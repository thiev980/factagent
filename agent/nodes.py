"""
FactAgent – Agent Nodes
=======================
Jeder Node ist ein eigenständiger Schritt im Agentic Workflow.
Nodes lesen vom State, führen eine Aktion aus (LLM-Call, Tool-Call),
und schreiben das Ergebnis zurück in den State.

AI-Engineering-Pattern: Agentic Workflow (ReAct)
- Jeder Schritt hat eine klare Verantwortung
- Schritte sind modular und testbar
- Der Graph orchestriert die Reihenfolge
"""

import json
import logging
from typing import Any

from anthropic import Anthropic

from agent.models import (
    AgentState,
    ClaimDecomposition,
    ClaimType,
    FactCheckResult,
    SubClaimVerdict,
)
from agent.prompts import (
    CLAIM_DECOMPOSER_SYSTEM,
    CLAIM_DECOMPOSER_USER,
    EVIDENCE_EVALUATOR_SYSTEM,
    EVIDENCE_EVALUATOR_USER,
    VERDICT_SYNTHESIZER_SYSTEM,
    VERDICT_SYNTHESIZER_USER,
)
from agent.tools import format_search_results_for_prompt, search_evidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_anthropic_client() -> Anthropic:
    """Erstellt einen Anthropic-Client (API-Key aus Umgebung)."""
    return Anthropic()


def call_claude_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> Any:
    """
    Ruft die Claude API auf und erzwingt strukturierte JSON-Ausgabe.
    
    AI-Engineering-Pattern: Structured Output
    - Wir geben dem LLM das JSON-Schema mit
    - Das LLM antwortet direkt im gewünschten Format
    - Pydantic validiert die Antwort
    
    Args:
        system_prompt: System-Prompt mit Rollenanweisung
        user_prompt: Der eigentliche Task
        response_model: Pydantic-Model für die Ausgabe
        model: Claude-Modell (Standard: Sonnet für gutes Preis/Leistung)
        max_tokens: Max. Tokens in der Antwort
    
    Returns:
        Validierte Pydantic-Model-Instanz
    """
    client = get_anthropic_client()

    # JSON-Schema aus dem Pydantic-Model generieren
    schema = response_model.model_json_schema()

    # System-Prompt um Schema-Anweisung ergänzen
    full_system = (
        f"{system_prompt}\n\n"
        f"## Ausgabeformat (JSON-Schema):\n"
        f"```json\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n```"
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=full_system,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    # Antworttext extrahieren
    raw_text = response.content[0].text.strip()

    # JSON aus der Antwort extrahieren (manchmal in ```json ... ``` gewrappt)
    if raw_text.startswith("```"):
        # Code-Block entfernen
        lines = raw_text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        raw_text = "\n".join(json_lines)

    # Parsen und validieren
    parsed = json.loads(raw_text)
    return response_model.model_validate(parsed)


# ---------------------------------------------------------------------------
# Node 1: Claim Decomposer
# ---------------------------------------------------------------------------

def decompose_claim(state: dict) -> dict:
    """
    Zerlegt die Behauptung in überprüfbare Teilaussagen.
    
    Input:  state["claim"] (str)
    Output: state["decomposition"] (ClaimDecomposition)
    """
    claim = state["claim"]
    logger.info(f"📝 Zerlege Behauptung: {claim[:80]}...")

    try:
        decomposition = call_claude_structured(
            system_prompt=CLAIM_DECOMPOSER_SYSTEM,
            user_prompt=CLAIM_DECOMPOSER_USER.format(claim=claim),
            response_model=ClaimDecomposition,
        )

        logger.info(
            f"✅ Zerlegung: {len(decomposition.sub_claims)} Teilaussagen, "
            f"Typ: {decomposition.claim_type}"
        )

        return {"decomposition": decomposition}

    except Exception as e:
        logger.error(f"❌ Fehler bei Claim Decomposition: {e}")
        return {"error": f"Fehler bei der Zerlegung: {e}"}


# ---------------------------------------------------------------------------
# Node 2: Evidence Retriever
# ---------------------------------------------------------------------------

def retrieve_evidence(state: dict) -> dict:
    """
    Sucht im Web nach Evidenz für jede Teilaussage.
    
    Input:  state["decomposition"] (ClaimDecomposition)
    Output: state["search_results"] (dict: sub_claim → results)
    """
    decomposition = state.get("decomposition")
    if not decomposition:
        return {"error": "Keine Zerlegung vorhanden"}

    # Bei Meinungsäusserungen: Kurzschluss
    if decomposition.claim_type == ClaimType.OPINION:
        logger.info("💭 Behauptung ist eine Meinung – begrenzte Faktenprüfung möglich")

    search_results = {}
    for sub_claim in decomposition.sub_claims:
        logger.info(f"🔍 Suche Evidenz für: {sub_claim.claim[:60]}...")
        results = search_evidence(sub_claim.search_queries)
        search_results[sub_claim.claim] = results
        logger.info(f"   → {len(results)} Quellen gefunden")

    return {"search_results": search_results}


# ---------------------------------------------------------------------------
# Node 3: Evidence Evaluator
# ---------------------------------------------------------------------------

def evaluate_evidence(state: dict) -> dict:
    """
    Bewertet die Evidenz und erstellt Verdicts für jede Teilaussage.
    
    Input:  state["decomposition"] + state["search_results"]
    Output: state["sub_verdicts"] (list[SubClaimVerdict])
    """
    decomposition = state.get("decomposition")
    search_results = state.get("search_results", {})

    if not decomposition:
        return {"error": "Keine Zerlegung vorhanden"}

    sub_verdicts = []

    for sub_claim in decomposition.sub_claims:
        claim_text = sub_claim.claim
        results = search_results.get(claim_text, [])

        logger.info(f"⚖️ Bewerte: {claim_text[:60]}...")

        if not results:
            # Keine Ergebnisse → unverifiable
            from agent.models import Verdict, Source
            sub_verdicts.append(SubClaimVerdict(
                claim=claim_text,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.1,
                evidence=[],
                reasoning="Keine relevanten Quellen gefunden.",
            ))
            continue

        try:
            formatted_results = format_search_results_for_prompt(results)

            verdict = call_claude_structured(
                system_prompt=EVIDENCE_EVALUATOR_SYSTEM,
                user_prompt=EVIDENCE_EVALUATOR_USER.format(
                    sub_claim=claim_text,
                    search_results=formatted_results,
                ),
                response_model=SubClaimVerdict,
            )

            sub_verdicts.append(verdict)
            logger.info(f"   → Verdikt: {verdict.verdict} (Konfidenz: {verdict.confidence})")

        except Exception as e:
            logger.error(f"❌ Fehler bei Bewertung von '{claim_text[:40]}': {e}")
            from agent.models import Verdict
            sub_verdicts.append(SubClaimVerdict(
                claim=claim_text,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=f"Bewertung fehlgeschlagen: {e}",
            ))

    return {"sub_verdicts": sub_verdicts}


# ---------------------------------------------------------------------------
# Node 4: Verdict Synthesizer
# ---------------------------------------------------------------------------

def synthesize_verdict(state: dict) -> dict:
    """
    Fasst alle Teilbewertungen zu einem Gesamtverdikt zusammen.
    
    Input:  state["claim"] + state["sub_verdicts"]
    Output: state["final_result"] (FactCheckResult)
    """
    claim = state["claim"]
    sub_verdicts = state.get("sub_verdicts", [])

    if not sub_verdicts:
        return {"error": "Keine Einzelbewertungen vorhanden"}

    logger.info("📊 Erstelle Gesamtverdikt...")

    # Einzelverdicts als JSON für den Prompt formatieren
    verdicts_json = json.dumps(
        [v.model_dump() for v in sub_verdicts],
        indent=2,
        ensure_ascii=False,
    )

    try:
        result = call_claude_structured(
            system_prompt=VERDICT_SYNTHESIZER_SYSTEM,
            user_prompt=VERDICT_SYNTHESIZER_USER.format(
                original_claim=claim,
                sub_verdicts=verdicts_json,
            ),
            response_model=FactCheckResult,
        )

        logger.info(
            f"✅ Gesamtverdikt: {result.overall_verdict} "
            f"(Konfidenz: {result.confidence})"
        )

        return {"final_result": result}

    except Exception as e:
        logger.error(f"❌ Fehler bei Gesamtverdikt: {e}")
        return {"error": f"Fehler bei der Zusammenfassung: {e}"}
