"""
FactAgent – Agent Nodes (v2: Async + Streaming)
================================================
Jeder Node ist ein eigenständiger Schritt im Agentic Workflow.
Nodes lesen vom State, führen eine Aktion aus (LLM-Call, Tool-Call),
und schreiben das Ergebnis zurück in den State.

AI-Engineering-Patterns:
- Agentic Workflow (ReAct)
- Streaming (async Token-by-Token-Ausgabe)

Update v2:
- Alle Nodes sind jetzt async (für Chainlit-Kompatibilität)
- Structured Output Calls streamen intern (für Fortschrittsanzeige)
- Neuer on_token Callback für Live-Updates in der UI
- stream_claude_text() Generator für die finale Zusammenfassung
"""

import json
import logging
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic

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

def get_async_client() -> AsyncAnthropic:
    """Erstellt einen async Anthropic-Client (API-Key aus Umgebung)."""
    return AsyncAnthropic()


def _extract_json(raw_text: str) -> str:
    """Extrahiert JSON aus einem LLM-Response (entfernt ```json``` Wrapper)."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
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
        return "\n".join(json_lines)
    return raw_text


def _repair_json(raw: str) -> str:
    """
    Versucht kaputtes JSON zu reparieren.
    
    Häufige Probleme bei LLM-Output:
    - Unescapte Anführungszeichen in Strings
    - Trailing Commas
    - Fehlende Klammern am Ende
    - Newlines in Strings
    - Einfache statt doppelte Anführungszeichen
    """
    import re

    # Schritt 1: Einfache Anführungszeichen durch doppelte ersetzen
    # (nur als äussere String-Delimiter, nicht innerhalb von Strings)
    if raw.count("'") > raw.count('"'):
        raw = raw.replace("'", '"')

    # Schritt 2: Trailing commas vor } oder ] entfernen
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    # Schritt 3: Versuche einfaches Parsen
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Schritt 4: Unescapte Anführungszeichen in String-Werten fixen
    # Strategie: Zeilenweise durch den JSON-Text gehen und
    # Anführungszeichen innerhalb von String-Werten escapen
    raw = _fix_unescaped_quotes(raw)

    # Schritt 5: Unescapte Newlines in Strings fixen
    raw = _fix_newlines_in_strings(raw)

    # Schritt 6: Fehlende schliessende Klammern ergänzen (stack-basiert)
    bracket_stack = []
    in_str = False
    for j, ch in enumerate(raw):
        if ch == '\\' and in_str:
            continue
        if ch == '"' and (j == 0 or raw[j-1] != '\\'):
            in_str = not in_str
        if in_str:
            continue
        if ch in '{[':
            bracket_stack.append('}' if ch == '{' else ']')
        elif ch in '}]' and bracket_stack:
            bracket_stack.pop()

    # Stack rückwärts schliessen (innerste zuerst)
    if bracket_stack:
        raw += ''.join(reversed(bracket_stack))

    # Schritt 7: Nochmals trailing commas (nach anderen Fixes)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    return raw


def _fix_unescaped_quotes(raw: str) -> str:
    """
    Repariert unescapte Anführungszeichen innerhalb von JSON-String-Werten.
    
    z.B.: {"text": "Er sagte "Hallo" zu ihr"}
    →     {"text": "Er sagte \"Hallo\" zu ihr"}
    
    Strategie: Character-by-character durch den Text gehen und
    zwischen "strukturellen" und "inhaltlichen" Quotes unterscheiden.
    """
    result = []
    i = 0
    in_string = False
    
    while i < len(raw):
        char = raw[i]
        
        if char == '\\' and in_string:
            # Escaped character – übernehmen und nächstes Zeichen überspringen
            result.append(char)
            if i + 1 < len(raw):
                i += 1
                result.append(raw[i])
            i += 1
            continue
        
        if char == '"':
            if not in_string:
                # String beginnt
                in_string = True
                result.append(char)
            else:
                # Sind wir am Ende eines Strings?
                # Schaue voraus: nach einem String-Ende kommt
                # Whitespace + eines von: , } ] :
                rest = raw[i + 1:].lstrip()
                if not rest or rest[0] in ',}]:':
                    # Strukturelles Quote → String endet
                    in_string = False
                    result.append(char)
                else:
                    # Inhaltliches Quote → escapen
                    result.append('\\"')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def _fix_newlines_in_strings(raw: str) -> str:
    """
    Ersetzt echte Newlines innerhalb von JSON-Strings durch \\n.
    """
    result = []
    in_string = False
    i = 0
    
    while i < len(raw):
        char = raw[i]
        
        if char == '\\' and in_string and i + 1 < len(raw):
            result.append(char)
            i += 1
            result.append(raw[i])
            i += 1
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        if char == '\n' and in_string:
            result.append('\\n')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


async def call_claude_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    on_token: Any = None,
    max_retries: int = 2,
) -> Any:
    """
    Ruft die Claude API auf mit Streaming und erzwingt strukturierte JSON-Ausgabe.
    
    AI-Engineering-Pattern: Structured Output + Streaming + Retry
    - Tokens werden gestreamt (für Fortschrittsanzeige)
    - Vollständiger JSON-Response wird am Ende validiert
    - Bei Parse-Fehlern: JSON-Reparatur + automatischer Retry
    
    Args:
        system_prompt: System-Prompt mit Rollenanweisung
        user_prompt: Der eigentliche Task
        response_model: Pydantic-Model für die Ausgabe
        model: Claude-Modell
        max_tokens: Max. Tokens in der Antwort
        on_token: Optional async callback pro Token.
                  Signatur: async def on_token(token: str, accumulated: str)
        max_retries: Max. Anzahl Wiederholungsversuche bei Parse-Fehlern
    
    Returns:
        Validierte Pydantic-Model-Instanz
    """
    client = get_async_client()

    # JSON-Schema aus dem Pydantic-Model generieren
    schema = response_model.model_json_schema()

    full_system = (
        f"{system_prompt}\n\n"
        f"## Ausgabeformat (JSON-Schema):\n"
        f"```json\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n```\n\n"
        f"WICHTIG: Antworte ausschliesslich mit validem JSON. "
        f"Alle Anführungszeichen innerhalb von String-Werten "
        f'MÜSSEN mit \\" escaped werden. Beispiel: "Er sagte \\"Hallo\\" zu ihr"'
    )

    last_error = None

    for attempt in range(1 + max_retries):
        try:
            # Bei Retry: klareren Prompt
            messages = [{"role": "user", "content": user_prompt}]
            if attempt > 0:
                logger.info(f"🔄 Retry {attempt}/{max_retries} nach JSON-Fehler...")
                messages = [
                    {"role": "user", "content": (
                        f"{user_prompt}\n\n"
                        "WICHTIG: Antworte NUR mit validem JSON. "
                        "Alle Anführungszeichen innerhalb von String-Werten "
                        'MÜSSEN mit \\" escaped werden. Keine Erklärungen, nur JSON.'
                    )},
                ]

            # Streaming API Call
            accumulated = ""
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=full_system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    accumulated += text
                    if on_token:
                        await on_token(text, accumulated)

            # JSON extrahieren
            json_text = _extract_json(accumulated)

            # Erst normales Parsen versuchen
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                # JSON-Reparatur versuchen
                logger.warning(f"⚠️ JSON-Parse-Fehler, versuche Reparatur (Attempt {attempt + 1})")
                repaired = _repair_json(json_text)
                parsed = json.loads(repaired)

            return response_model.model_validate(parsed)

        except (json.JSONDecodeError, Exception) as e:
            last_error = e
            logger.warning(
                f"⚠️ Attempt {attempt + 1} fehlgeschlagen: {type(e).__name__}: {e}"
            )
            if attempt < max_retries:
                continue

    # Alle Versuche fehlgeschlagen
    raise last_error


async def stream_claude_text(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    Streamt eine Claude-Antwort Token für Token als async Generator.
    
    AI-Engineering-Pattern: Streaming
    - Kein Warten auf die komplette Antwort
    - User sieht Text in Echtzeit erscheinen
    - Bessere UX bei langen Antworten
    
    Yields:
        Einzelne Text-Tokens
    """
    client = get_async_client()

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Node 1: Claim Decomposer (async)
# ---------------------------------------------------------------------------

async def decompose_claim(state: dict, on_token=None) -> dict:
    """
    Zerlegt die Behauptung in überprüfbare Teilaussagen.
    
    Input:  state["claim"] (str)
    Output: state["decomposition"] (ClaimDecomposition)
    """
    claim = state["claim"]
    logger.info(f"📝 Zerlege Behauptung: {claim[:80]}...")

    try:
        decomposition = await call_claude_structured(
            system_prompt=CLAIM_DECOMPOSER_SYSTEM,
            user_prompt=CLAIM_DECOMPOSER_USER.format(claim=claim),
            response_model=ClaimDecomposition,
            on_token=on_token,
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
# Node 2: Evidence Retriever (sync I/O, async wrapper)
# ---------------------------------------------------------------------------

async def retrieve_evidence(state: dict) -> dict:
    """
    Sucht im Web nach Evidenz für jede Teilaussage.
    
    Input:  state["decomposition"] (ClaimDecomposition)
    Output: state["search_results"] (dict: sub_claim → results)
    
    Hinweis: Tavily-Client ist synchron, aber die Funktion ist async
    damit sie in den Chainlit-Event-Loop passt.
    """
    decomposition = state.get("decomposition")
    if not decomposition:
        return {"error": "Keine Zerlegung vorhanden"}

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
# Node 3: Evidence Evaluator (async + streaming)
# ---------------------------------------------------------------------------

async def evaluate_evidence(state: dict, on_token=None) -> dict:
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

            verdict = await call_claude_structured(
                system_prompt=EVIDENCE_EVALUATOR_SYSTEM,
                user_prompt=EVIDENCE_EVALUATOR_USER.format(
                    sub_claim=claim_text,
                    search_results=formatted_results,
                ),
                response_model=SubClaimVerdict,
                on_token=on_token,
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
# Node 4: Verdict Synthesizer (async + streaming + HITL)
# ---------------------------------------------------------------------------

def _format_human_feedback(state: dict) -> str:
    """Formatiert Human Feedback für den Synthesizer-Prompt."""
    feedback = state.get("human_feedback")

    if not feedback or not feedback.reviewed:
        return "Keine menschliche Überprüfung – basiere das Verdikt auf den AI-Bewertungen."

    parts = []

    if feedback.general_comment:
        parts.append(f"Allgemeiner Kommentar des Users: {feedback.general_comment}")

    corrections = []
    for fb in feedback.sub_claim_feedback:
        if fb.corrected_verdict:
            line = f"  - «{fb.claim}»: AI sagte etwas anderes, User korrigiert zu → {fb.corrected_verdict.value}"
            if fb.user_comment:
                line += f" (Begründung: {fb.user_comment})"
            corrections.append(line)
        elif fb.user_comment:
            corrections.append(f"  - «{fb.claim}»: User-Kommentar: {fb.user_comment}")

    if corrections:
        parts.append("Korrekturen des Users:\n" + "\n".join(corrections))
    else:
        parts.append("Der User hat alle AI-Bewertungen bestätigt (keine Korrekturen).")

    return "\n".join(parts) if parts else "Keine Korrekturen."


async def synthesize_verdict(state: dict, on_token=None) -> dict:
    """
    Fasst alle Teilbewertungen zu einem Gesamtverdikt zusammen.
    Berücksichtigt Human-in-the-Loop Feedback, falls vorhanden.
    
    AI-Engineering-Pattern: Human-in-the-Loop
    - Menschliche Korrekturen werden stark gewichtet
    - Transparente Dokumentation von AI- vs. User-Bewertung
    
    Input:  state["claim"] + state["sub_verdicts"] + state["human_feedback"]
    Output: state["final_result"] (FactCheckResult)
    """
    claim = state["claim"]
    sub_verdicts = state.get("sub_verdicts", [])

    if not sub_verdicts:
        return {"error": "Keine Einzelbewertungen vorhanden"}

    # Human Feedback in die sub_verdicts einarbeiten
    feedback = state.get("human_feedback")
    if feedback and feedback.reviewed:
        for fb in feedback.sub_claim_feedback:
            if fb.corrected_verdict:
                # Finde die passende sub_verdict und aktualisiere sie
                for sv in sub_verdicts:
                    if sv.claim == fb.claim:
                        logger.info(
                            f"👤 User-Korrektur: '{sv.claim[:40]}' "
                            f"{sv.verdict} → {fb.corrected_verdict}"
                        )
                        sv.verdict = fb.corrected_verdict
                        if fb.user_comment:
                            sv.reasoning += f" [User-Korrektur: {fb.user_comment}]"

    logger.info("📊 Erstelle Gesamtverdikt...")

    verdicts_json = json.dumps(
        [v.model_dump() for v in sub_verdicts],
        indent=2,
        ensure_ascii=False,
    )

    human_feedback_text = _format_human_feedback(state)

    try:
        result = await call_claude_structured(
            system_prompt=VERDICT_SYNTHESIZER_SYSTEM,
            user_prompt=VERDICT_SYNTHESIZER_USER.format(
                original_claim=claim,
                sub_verdicts=verdicts_json,
                human_feedback=human_feedback_text,
            ),
            response_model=FactCheckResult,
            on_token=on_token,
        )

        logger.info(
            f"✅ Gesamtverdikt: {result.overall_verdict} "
            f"(Konfidenz: {result.confidence})"
        )

        return {"final_result": result}

    except Exception as e:
        logger.error(f"❌ Fehler bei Gesamtverdikt: {e}")
        return {"error": f"Fehler bei der Zusammenfassung: {e}"}
