"""
FactAgent – Chainlit Web App (v4: Streaming + HITL + Historische Claims)
========================================================================
Das Frontend: Eine Chat-UI, in der User Behauptungen eingeben
und den Faktencheck-Prozess live verfolgen können.

Update v4: Historische Claims
- SQLite-Datenbank speichert alle abgeschlossenen Checks
- Vor jedem neuen Check: Suche nach ähnlichen früheren Behauptungen
- User kann vorheriges Ergebnis übernehmen oder neu prüfen
- /history Befehl zeigt die letzten Checks
- /stats zeigt Statistiken

AI-Engineering-Patterns:
- Human-in-the-Loop
- Streaming
- Caching / Knowledge Base
"""

import json
import logging
import time

import chainlit as cl
from dotenv import load_dotenv

from agent.models import (
    FactCheckResult,
    HumanFeedback,
    SubClaimFeedback,
    Verdict,
)
from agent.nodes import (
    decompose_claim,
    retrieve_evidence,
    evaluate_evidence,
    synthesize_verdict,
    stream_claude_text,
)
from agent.prompts import (
    STREAMING_SUMMARY_SYSTEM,
    STREAMING_SUMMARY_USER,
)
from agent.database import (
    init_db,
    store_fact_check,
    find_exact_claim,
    find_similar_claims,
    get_recent_checks,
    get_stats,
)
from agent.source_graph import generate_graph_html

# .env laden (API-Keys)
load_dotenv()

# Datenbank initialisieren
init_db()

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

# Mapping von Action-Label zurück zu Verdict Enum
VERDICT_FROM_LABEL = {v: k for k, v in VERDICT_LABEL_DE.items()}


def format_confidence_bar(confidence: float) -> str:
    """Erstellt eine visuelle Konfidenz-Anzeige."""
    filled = int(confidence * 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    return f"`{bar}` {confidence:.0%}"


def format_result_header(result: FactCheckResult) -> str:
    """Formatiert den Verdikt-Header (Zusammenfassung wird gestreamt)."""
    emoji = VERDICT_EMOJI.get(result.overall_verdict, "❓")
    label = VERDICT_LABEL_DE.get(result.overall_verdict, result.overall_verdict)

    lines = [
        f"## {emoji} Gesamtverdikt: {label}",
        f"**Konfidenz:** {format_confidence_bar(result.confidence)}",
        "",
    ]

    return "\n".join(lines)


def format_result_details(result: FactCheckResult) -> str:
    """Formatiert die Quellen für das Endergebnis (Einzelbewertungen werden separat angezeigt)."""
    lines = ["", "---"]

    if result.key_sources:
        lines.append("### Quellen")
        for source in result.key_sources:
            credibility_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                source.credibility, "⚪"
            )
            lines.append(
                f"- {credibility_icon} [{source.title}]({source.url})"
            )

    return "\n".join(lines)


def _make_token_counter():
    """Erstellt einen Token-Zähler-Callback für Step-Updates."""
    state = {"count": 0}

    async def on_token(token: str, accumulated: str):
        state["count"] += 1

    return on_token, state


def _get_action_payload(res) -> dict:
    """
    Extrahiert das Payload aus einer AskActionMessage-Antwort.
    Kompatibel mit verschiedenen Chainlit-Versionen (dict oder Objekt).
    """
    if res is None:
        return {}
    # Chainlit 2.x: res kann ein Action-Objekt oder ein dict sein
    if isinstance(res, dict):
        payload = res.get("payload", {})
    elif hasattr(res, "payload"):
        payload = res.payload
    else:
        return {}
    # payload kann selbst ein dict oder string sein
    if isinstance(payload, str):
        return {"action": payload}
    return payload if isinstance(payload, dict) else {}


def format_sub_verdicts_for_review(sub_verdicts) -> str:
    """Formatiert die Teilurteile für die Nutzer-Überprüfung."""
    lines = ["### ⚖️ AI-Bewertung der Teilaussagen\n"]

    for i, sv in enumerate(sub_verdicts, 1):
        emoji = VERDICT_EMOJI.get(sv.verdict, "❓")
        label = VERDICT_LABEL_DE.get(sv.verdict, sv.verdict)
        lines.extend([
            f"**{i}. {sv.claim}**",
            f"   {emoji} **{label}** · Konfidenz: {sv.confidence:.0%}",
            f"   *{sv.reasoning}*",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-in-the-Loop: Interaktive Überprüfung
# ---------------------------------------------------------------------------

async def collect_human_feedback(sub_verdicts) -> HumanFeedback:
    """
    Zeigt dem User die AI-Bewertungen und sammelt Korrekturen ein.
    
    AI-Engineering-Pattern: Human-in-the-Loop
    - User sieht jede Teilbewertung
    - Kann einzelne Verdikts überschreiben
    - Kann Kommentare/Kontext hinzufügen
    - Alles wird strukturiert als HumanFeedback erfasst
    """
    sub_claim_feedback = []

    for i, sv in enumerate(sub_verdicts, 1):
        emoji = VERDICT_EMOJI.get(sv.verdict, "❓")
        label = VERDICT_LABEL_DE.get(sv.verdict, sv.verdict)

        # Für jede Teilaussage: Akzeptieren oder Korrigieren?
        actions = [
            cl.Action(
                name="accept",
                label=f"✅ {label} akzeptieren",
                payload={"action": "accept"},
            ),
            cl.Action(
                name="correct",
                label="✏️ Korrigieren",
                payload={"action": "correct"},
            ),
        ]

        res = await cl.AskActionMessage(
            content=(
                f"**Teilaussage {i}/{len(sub_verdicts)}:**\n"
                f"*«{sv.claim}»*\n\n"
                f"AI-Bewertung: {emoji} **{label}** ({sv.confidence:.0%})\n"
                f"Begründung: *{sv.reasoning}*"
            ),
            actions=actions,
            timeout=300,
        ).send()

        if res and _get_action_payload(res).get("action") == "correct":
            # User will korrigieren → Welches Verdikt?
            verdict_actions = [
                cl.Action(name="v", label=f"{VERDICT_EMOJI[v]} {VERDICT_LABEL_DE[v]}", payload={"verdict": v.value})
                for v in Verdict
            ]

            verdict_res = await cl.AskActionMessage(
                content=f"Welches Verdikt für *«{sv.claim}»*?",
                actions=verdict_actions,
                timeout=300,
            ).send()

            corrected_verdict = None
            if verdict_res:
                payload = _get_action_payload(verdict_res)
                verdict_val = payload.get("verdict")
                if verdict_val:
                    corrected_verdict = Verdict(verdict_val)

            # Optional: User-Kommentar
            comment_res = await cl.AskUserMessage(
                content=(
                    "Möchtest du einen Kommentar oder Kontext hinzufügen? "
                    "(Einfach Enter für keinen Kommentar)"
                ),
                timeout=300,
            ).send()

            user_comment = None
            if comment_res:
                # Kompatibel mit verschiedenen Chainlit-Versionen
                text = ""
                if isinstance(comment_res, dict):
                    text = comment_res.get("output", "")
                elif hasattr(comment_res, "output"):
                    text = comment_res.output or ""
                elif hasattr(comment_res, "content"):
                    text = comment_res.content or ""
                user_comment = text.strip() or None

            sub_claim_feedback.append(SubClaimFeedback(
                claim=sv.claim,
                corrected_verdict=corrected_verdict,
                user_comment=user_comment,
            ))

            emoji_new = VERDICT_EMOJI.get(corrected_verdict, "❓") if corrected_verdict else emoji
            label_new = VERDICT_LABEL_DE.get(corrected_verdict, "?") if corrected_verdict else label
            await cl.Message(
                content=f"👤 Korrektur erfasst: {emoji_new} **{label_new}**"
                + (f" – *{user_comment}*" if user_comment else ""),
            ).send()

        else:
            # Akzeptiert
            sub_claim_feedback.append(SubClaimFeedback(
                claim=sv.claim,
                corrected_verdict=None,
                user_comment=None,
            ))

    # Optional: Allgemeiner Kommentar
    general_res = await cl.AskUserMessage(
        content=(
            "Hast du einen allgemeinen Kommentar zur Behauptung? "
            "(Einfach Enter zum Überspringen)"
        ),
        timeout=300,
    ).send()

    general_comment = None
    if general_res:
        text = ""
        if isinstance(general_res, dict):
            text = general_res.get("output", "")
        elif hasattr(general_res, "output"):
            text = general_res.output or ""
        elif hasattr(general_res, "content"):
            text = general_res.content or ""
        general_comment = text.strip() or None

    return HumanFeedback(
        reviewed=True,
        sub_claim_feedback=sub_claim_feedback,
        general_comment=general_comment,
    )


# ---------------------------------------------------------------------------
# Chainlit Event Handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    """Wird beim Start einer neuen Chat-Session aufgerufen."""
    # Stats anzeigen
    stats = get_stats()
    stats_line = ""
    if stats["total_checks"] > 0:
        stats_line = (
            f"\n📊 *{stats['total_checks']} Behauptungen bereits geprüft "
            f"({stats['human_reviewed']} davon manuell überprüft)*\n"
        )

    await cl.Message(
        content=(
            "## 🔍 FactAgent – Faktencheck-Assistent\n\n"
            "Gib eine Behauptung ein, und ich überprüfe sie für dich.\n"
            + stats_line +
            "\n**Beispiele:**\n"
            "- *«Die Schweiz hat die höchste Einwanderungsrate in Europa.»*\n"
            "- *«ChatGPT wurde von Google entwickelt.»*\n"
            "- *«Die Erde ist 4,5 Milliarden Jahre alt.»*\n\n"
            "**Befehle:**\n"
            "- `/history` – Letzte Faktenchecks anzeigen\n"
            "- `/stats` – Statistiken anzeigen\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Wird bei jeder User-Nachricht aufgerufen."""
    claim = message.content.strip()

    if not claim:
        await cl.Message(content="Bitte gib eine Behauptung ein.").send()
        return

    # ---- Befehle verarbeiten ----
    if claim.lower() == "/history":
        recent = get_recent_checks(limit=10)
        if not recent:
            await cl.Message(content="📋 Noch keine Faktenchecks durchgeführt.").send()
            return

        lines = ["## 📋 Letzte Faktenchecks\n"]
        for check in recent:
            emoji = VERDICT_EMOJI.get(Verdict(check["verdict"]), "❓")
            reviewed = " 👤" if check["human_reviewed"] else ""
            date = check["created_at"][:10]
            lines.append(
                f"- {emoji} **{check['claim'][:80]}** "
                f"({check['confidence']:.0%}) – {date}{reviewed}"
            )
        await cl.Message(content="\n".join(lines)).send()
        return

    if claim.lower() == "/stats":
        stats = get_stats()
        verdicts_text = "\n".join(
            f"  - {VERDICT_EMOJI.get(Verdict(v), '❓')} {VERDICT_LABEL_DE.get(Verdict(v), v)}: {c}"
            for v, c in stats["by_verdict"].items()
        ) if stats["by_verdict"] else "  Noch keine Daten."

        await cl.Message(
            content=(
                f"## 📊 FactAgent Statistiken\n\n"
                f"**Geprüfte Behauptungen:** {stats['total_checks']}\n"
                f"**Davon manuell überprüft:** {stats['human_reviewed']}\n\n"
                f"**Verdikts:**\n{verdicts_text}"
            )
        ).send()
        return

    # ---- Datenbank-Check: Wurde diese Behauptung schon geprüft? ----
    exact_match = find_exact_claim(claim)
    if exact_match and exact_match["result"]:
        prev = exact_match
        emoji = VERDICT_EMOJI.get(Verdict(prev["verdict"]), "❓")
        label = VERDICT_LABEL_DE.get(Verdict(prev["verdict"]), prev["verdict"])
        reviewed_tag = " (👤 manuell überprüft)" if prev["human_reviewed"] else ""

        reuse_action = await cl.AskActionMessage(
            content=(
                f"### 💾 Diese Behauptung wurde bereits geprüft!\n\n"
                f"**Vorheriges Ergebnis:** {emoji} {label} "
                f"({prev['confidence']:.0%}){reviewed_tag}\n"
                f"*Geprüft am {prev['created_at'][:10]}*\n\n"
                f"Möchtest du das vorherige Ergebnis verwenden oder neu prüfen?"
            ),
            actions=[
                cl.Action(name="reuse", label="✅ Vorheriges verwenden", payload={"action": "reuse"}),
                cl.Action(name="recheck", label="🔄 Neu prüfen", payload={"action": "recheck"}),
            ],
            timeout=300,
        ).send()

        if reuse_action and _get_action_payload(reuse_action).get("action") == "reuse":
            # Vorheriges Ergebnis anzeigen
            result = prev["result"]
            header = format_result_header(result)
            details = format_result_details(result)
            await cl.Message(
                content=header + "\n" + result.summary + details + "\n\n*💾 Aus der Datenbank geladen*"
            ).send()
            return

    # Ähnliche Claims suchen (nicht-exakt)
    if not exact_match:
        similar = find_similar_claims(claim, limit=3)
        if similar:
            lines = ["### 🔎 Ähnliche frühere Checks gefunden:\n"]
            for s in similar:
                emoji = VERDICT_EMOJI.get(Verdict(s["verdict"]), "❓")
                lines.append(f"- {emoji} *«{s['claim'][:80]}»* ({s['confidence']:.0%})")

            await cl.Message(content="\n".join(lines) + "\n\n*Starte trotzdem einen neuen Check...*").send()

    # ---- Timer starten (für DB-Speicherung) ----
    check_start_time = time.time()

    result_state = {"claim": claim}

    # ---- Schritt 1: Zerlegung ----
    async with cl.Step(name="🧩 Behauptung zerlegen", type="tool") as step:
        step.input = claim
        token_cb, token_state = _make_token_counter()

        decomp_result = await decompose_claim(result_state, on_token=token_cb)
        result_state.update(decomp_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(content=f"❌ {result_state['error']}").send()
            return

        decomp = result_state["decomposition"]
        sub_claims_text = "\n".join(
            f"  {i}. {sc.claim}" for i, sc in enumerate(decomp.sub_claims, 1)
        )
        step.output = (
            f"Typ: {decomp.claim_type.value} · "
            f"{token_state['count']} Tokens generiert\n"
            f"Teilaussagen:\n{sub_claims_text}"
        )

    # ---- Schritt 2: Evidenz suchen ----
    async with cl.Step(name="🔍 Quellen suchen", type="tool") as step:
        total_queries = sum(
            len(sc.search_queries) for sc in decomp.sub_claims
        )
        step.input = f"{total_queries} Suchanfragen für {len(decomp.sub_claims)} Teilaussagen"

        evidence_result = await retrieve_evidence(result_state)
        result_state.update(evidence_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(content=f"❌ {result_state['error']}").send()
            return

        total_sources = sum(
            len(v) for v in result_state["search_results"].values()
        )
        step.output = f"{total_sources} Quellen gefunden"

    # ---- Schritt 3: Evidenz bewerten ----
    async with cl.Step(name="⚖️ Quellen bewerten", type="tool") as step:
        step.input = f"Bewerte Evidenz für {len(decomp.sub_claims)} Teilaussagen"
        token_cb, token_state = _make_token_counter()

        eval_result = await evaluate_evidence(result_state, on_token=token_cb)
        result_state.update(eval_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(content=f"❌ {result_state['error']}").send()
            return

        verdicts_summary = "\n".join(
            f"  {sv.verdict.value} ({sv.confidence:.0%}): {sv.claim[:50]}"
            for sv in result_state["sub_verdicts"]
        )
        step.output = f"{token_state['count']} Tokens · Bewertungen:\n{verdicts_summary}"

    # ---- HUMAN-IN-THE-LOOP: Überprüfung ----
    # Zeige die Ergebnisse und frage den User
    review_text = format_sub_verdicts_for_review(result_state["sub_verdicts"])
    await cl.Message(content=review_text).send()

    review_action = await cl.AskActionMessage(
        content=(
            "**Möchtest du die Bewertungen überprüfen?**\n\n"
            "Du kannst einzelne Verdikts korrigieren oder Kontext hinzufügen, "
            "bevor das Gesamtverdikt erstellt wird."
        ),
        actions=[
            cl.Action(
                name="accept",
                label="✅ Alle akzeptieren & weiter",
                payload={"action": "accept"},
            ),
            cl.Action(
                name="review",
                label="✏️ Einzeln überprüfen",
                payload={"action": "review"},
            ),
        ],
        timeout=300,
    ).send()

    # Feedback verarbeiten
    human_feedback = HumanFeedback(reviewed=False)

    if review_action and _get_action_payload(review_action).get("action") == "review":
        await cl.Message(
            content="👤 **Überprüfungsmodus** – Gehe jede Teilaussage durch:"
        ).send()
        human_feedback = await collect_human_feedback(result_state["sub_verdicts"])

        # Zusammenfassung des Feedbacks
        corrections = sum(
            1 for fb in human_feedback.sub_claim_feedback if fb.corrected_verdict
        )
        await cl.Message(
            content=(
                f"✅ Überprüfung abgeschlossen. "
                f"**{corrections} Korrektur(en)** erfasst. "
                f"Erstelle jetzt das Gesamtverdikt..."
            )
        ).send()
    else:
        await cl.Message(
            content="✅ Bewertungen akzeptiert. Erstelle Gesamtverdikt..."
        ).send()

    result_state["human_feedback"] = human_feedback

    # ---- Schritt 4: Gesamtverdikt (mit Feedback) ----
    async with cl.Step(name="📊 Gesamtverdikt erstellen", type="tool") as step:
        feedback_note = ""
        if human_feedback.reviewed:
            corrections = sum(
                1 for fb in human_feedback.sub_claim_feedback if fb.corrected_verdict
            )
            feedback_note = f" (mit {corrections} User-Korrektur(en))"
        step.input = f"Synthese der Einzelbewertungen{feedback_note}"
        token_cb, token_state = _make_token_counter()

        synth_result = await synthesize_verdict(result_state, on_token=token_cb)
        result_state.update(synth_result)

        if result_state.get("error"):
            step.output = f"Fehler: {result_state['error']}"
            await cl.Message(content=f"❌ {result_state['error']}").send()
            return

        step.output = (
            f"Verdikt: {result_state['final_result'].overall_verdict.value} "
            f"({result_state['final_result'].confidence:.0%}) · "
            f"{token_state['count']} Tokens"
        )

    # ---- Ergebnis streamen ----
    final_result = result_state["final_result"]

    # 1) Header sofort anzeigen
    header = format_result_header(final_result)
    msg = cl.Message(content=header)
    await msg.send()

    # 2) Human Review Note für Streaming Summary
    if human_feedback.reviewed:
        corrections = sum(
            1 for fb in human_feedback.sub_claim_feedback if fb.corrected_verdict
        )
        human_review_note = (
            f"Dieses Verdikt wurde vom User überprüft. "
            f"{corrections} Korrektur(en) wurden eingearbeitet. "
            f"Erwähne dies kurz in der Zusammenfassung."
        )
    else:
        human_review_note = "Keine menschliche Überprüfung erfolgt."

    # 3) Zusammenfassung Token-by-Token streamen
    sub_verdicts_text = "\n".join(
        f"- {sv.claim}: {sv.verdict.value} ({sv.confidence:.0%}) – {sv.reasoning}"
        for sv in final_result.sub_verdicts
    )
    sources_text = "\n".join(
        f"- [{s.title}]({s.url}) (Glaubwürdigkeit: {s.credibility.value})"
        for s in final_result.key_sources
    )

    summary_prompt = STREAMING_SUMMARY_USER.format(
        claim=final_result.original_claim,
        verdict=VERDICT_LABEL_DE.get(final_result.overall_verdict, "?"),
        confidence=final_result.confidence,
        sub_verdicts_text=sub_verdicts_text,
        sources_text=sources_text,
        human_review_note=human_review_note,
    )

    async for token in stream_claude_text(
        system_prompt=STREAMING_SUMMARY_SYSTEM,
        user_prompt=summary_prompt,
    ):
        await msg.stream_token(token)

    # 4) Details anhängen
    details = format_result_details(final_result)
    await msg.stream_token(details)

    await msg.update()

    # ---- In Datenbank speichern ----
    check_duration = time.time() - check_start_time
    try:
        db_id = store_fact_check(
            claim=claim,
            result=final_result,
            human_reviewed=human_feedback.reviewed,
            duration_seconds=check_duration,
        )
        await cl.Message(
            content=f"💾 *Ergebnis gespeichert (#{db_id}, {check_duration:.1f}s)*"
        ).send()
    except Exception as e:
        logger.error(f"❌ Fehler beim Speichern: {e}")
        await cl.Message(content=f"⚠️ *Speichern fehlgeschlagen: {e}*").send()

    # ---- Source Graph generieren ----
    try:
        graph_path = generate_graph_html(final_result, claim=claim)

        # Als Chainlit-Element einbetten (Link zum Öffnen)
        graph_element = cl.File(
            name="source_graph.html",
            path=graph_path,
            display="inline",
        )
        await cl.Message(
            content="### 🕸️ Source Graph\n\n"
            "Interaktive Visualisierung der Quellen und Teilaussagen. "
            "Lade die Datei herunter und öffne sie im Browser.\n\n"
            "- **Rechtecke** = Teilaussagen (Farbe = Verdikt)\n"
            "- **Kreise** = Quellen (Farbe = Glaubwürdigkeit, Grösse = Relevanz)\n"
            "- **Hover** für Details, **Doppelklick** auf Quelle öffnet URL",
            elements=[graph_element],
        ).send()

    except Exception as e:
        logger.error(f"❌ Source Graph Fehler: {e}")
        await cl.Message(
            content=f"⚠️ *Source Graph konnte nicht erstellt werden: {e}*"
        ).send()
