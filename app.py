"""
FactAgent – Chainlit Web App
=============================
Das Frontend: Eine Chat-UI, in der User Behauptungen eingeben
und den Faktencheck-Prozess live verfolgen können.

Chainlit bietet:
- Chat-Interface out of the box
- Streaming-Support
- Step-Visualisierung (der User sieht jeden Agent-Schritt)
- Deployment-ready
"""

import json
import logging

import chainlit as cl
from dotenv import load_dotenv

from agent.graph import build_fact_check_graph
from agent.models import FactCheckResult, Verdict

# .env laden (API-Keys)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict Styling
# ---------------------------------------------------------------------------

VERDICT_EMOJI = {
    Verdict.TRUE: "✅",
    Verdict.FALSE: "❌",
    Verdict.PARTIALLY_TRUE: "🟡",
    Verdict.MISLEADING: "⚠️",
    Verdict.UNVERIFIABLE: "❓",
}

VERDICT_LABEL_DE = {
    Verdict.TRUE: "Wahr",
    Verdict.FALSE: "Falsch",
    Verdict.PARTIALLY_TRUE: "Teilweise wahr",
    Verdict.MISLEADING: "Irreführend",
    Verdict.UNVERIFIABLE: "Nicht überprüfbar",
}


def format_confidence_bar(confidence: float) -> str:
    """Erstellt eine visuelle Konfidenz-Anzeige."""
    filled = int(confidence * 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    return f"`{bar}` {confidence:.0%}"


def format_result(result: FactCheckResult) -> str:
    """Formatiert das Ergebnis als Markdown für die Chat-Anzeige."""
    emoji = VERDICT_EMOJI.get(result.overall_verdict, "❓")
    label = VERDICT_LABEL_DE.get(result.overall_verdict, result.overall_verdict)

    lines = [
        f"## {emoji} Gesamtverdikt: {label}",
        f"**Konfidenz:** {format_confidence_bar(result.confidence)}",
        "",
        f"### Zusammenfassung",
        result.summary,
        "",
        "---",
        "### Einzelbewertungen",
    ]

    for i, sv in enumerate(result.sub_verdicts, 1):
        sv_emoji = VERDICT_EMOJI.get(sv.verdict, "❓")
        sv_label = VERDICT_LABEL_DE.get(sv.verdict, sv.verdict)
        lines.extend([
            f"**{i}. {sv.claim}**",
            f"  {sv_emoji} {sv_label} · Konfidenz: {sv.confidence:.0%}",
            f"  *{sv.reasoning}*",
            "",
        ])

    # Quellen
    if result.key_sources:
        lines.append("---")
        lines.append("### Quellen")
        for source in result.key_sources:
            credibility_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                source.credibility, "⚪"
            )
            lines.append(
                f"- {credibility_icon} [{source.title}]({source.url})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chainlit Event Handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    """Wird beim Start einer neuen Chat-Session aufgerufen."""
    await cl.Message(
        content=(
            "## 🔍 FactAgent – Faktencheck-Assistent\n\n"
            "Gib eine Behauptung ein, und ich überprüfe sie für dich.\n\n"
            "**Beispiele:**\n"
            "- *«Die Schweiz hat die höchste Einwanderungsrate in Europa.»*\n"
            "- *«ChatGPT wurde von Google entwickelt.»*\n"
            "- *«Die Erde ist 4,5 Milliarden Jahre alt.»*\n"
            "- *«Deutschland gibt mehr für Verteidigung aus als Frankreich.»*\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Wird bei jeder User-Nachricht aufgerufen."""
    claim = message.content.strip()

    if not claim:
        await cl.Message(content="Bitte gib eine Behauptung ein.").send()
        return

    # Graph kompilieren
    graph = build_fact_check_graph()

    # ---- Schritt 1: Zerlegung ----
    async with cl.Step(name="🧩 Behauptung zerlegen", type="tool") as step:
        step.input = claim
        result_state = {"claim": claim}

        # Wir führen den Graph schrittweise aus, um Steps zu zeigen
        # Alternativ: graph.invoke() für alles auf einmal

        from agent.nodes import decompose_claim
        decomp_result = decompose_claim(result_state)
        result_state.update(decomp_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(
                content=f"❌ {result_state['error']}"
            ).send()
            return

        decomp = result_state["decomposition"]
        sub_claims_text = "\n".join(
            f"  {i}. {sc.claim}" for i, sc in enumerate(decomp.sub_claims, 1)
        )
        step.output = (
            f"Typ: {decomp.claim_type.value}\n"
            f"Teilaussagen:\n{sub_claims_text}"
        )

    # ---- Schritt 2: Evidenz suchen ----
    async with cl.Step(name="🔍 Quellen suchen", type="tool") as step:
        total_queries = sum(
            len(sc.search_queries) for sc in decomp.sub_claims
        )
        step.input = f"{total_queries} Suchanfragen für {len(decomp.sub_claims)} Teilaussagen"

        from agent.nodes import retrieve_evidence
        evidence_result = retrieve_evidence(result_state)
        result_state.update(evidence_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(
                content=f"❌ {result_state['error']}"
            ).send()
            return

        total_sources = sum(
            len(v) for v in result_state["search_results"].values()
        )
        step.output = f"{total_sources} Quellen gefunden"

    # ---- Schritt 3: Evidenz bewerten ----
    async with cl.Step(name="⚖️ Quellen bewerten", type="tool") as step:
        step.input = f"Bewerte Evidenz für {len(decomp.sub_claims)} Teilaussagen"

        from agent.nodes import evaluate_evidence
        eval_result = evaluate_evidence(result_state)
        result_state.update(eval_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(
                content=f"❌ {result_state['error']}"
            ).send()
            return

        verdicts_summary = "\n".join(
            f"  {sv.verdict.value} ({sv.confidence:.0%}): {sv.claim[:50]}"
            for sv in result_state["sub_verdicts"]
        )
        step.output = verdicts_summary

    # ---- Schritt 4: Gesamtverdikt ----
    async with cl.Step(name="📊 Gesamtverdikt erstellen", type="tool") as step:
        step.input = "Synthese der Einzelbewertungen"

        from agent.nodes import synthesize_verdict
        synth_result = synthesize_verdict(result_state)
        result_state.update(synth_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(
                content=f"❌ {result_state['error']}"
            ).send()
            return

        step.output = (
            f"Verdikt: {result_state['final_result'].overall_verdict.value} "
            f"({result_state['final_result'].confidence:.0%})"
        )

    # ---- Ergebnis anzeigen ----
    final_result = result_state["final_result"]
    formatted = format_result(final_result)
    await cl.Message(content=formatted).send()
