"""
FactAgent – LangGraph Workflow
==============================
Hier wird der Agentic Workflow als Graph definiert.
LangGraph orchestriert die Reihenfolge der Schritte und
verwaltet den State zwischen den Nodes.

AI-Engineering-Pattern: Agentic Workflow / State Machine
- Klare Schritte mit definierten Ein-/Ausgaben
- Fehlerbehandlung an jedem Übergang
- Einfach erweiterbar (neue Nodes hinzufügen)
"""

import logging
from typing import TypedDict, Optional, Annotated

from langgraph.graph import StateGraph, END

from agent.models import (
    ClaimDecomposition,
    FactCheckResult,
    SubClaimVerdict,
)
from agent.nodes import (
    decompose_claim,
    retrieve_evidence,
    evaluate_evidence,
    synthesize_verdict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph State (TypedDict für LangGraph)
# ---------------------------------------------------------------------------
# LangGraph verwendet TypedDict statt Pydantic für den State.
# Die Pydantic-Models werden innerhalb des States als Werte genutzt.

class GraphState(TypedDict, total=False):
    """State, der durch den Workflow fliesst."""
    claim: str
    decomposition: Optional[ClaimDecomposition]
    search_results: dict
    sub_verdicts: list[SubClaimVerdict]
    final_result: Optional[FactCheckResult]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Routing-Logik
# ---------------------------------------------------------------------------

def should_continue_after_decomposition(state: GraphState) -> str:
    """
    Entscheidet nach der Zerlegung, ob weitergemacht werden soll.
    
    Routing-Logik:
    - Fehler → Ende
    - Reine Meinungsäusserung → Trotzdem weiter (aber mit Hinweis)
    - Alles andere → Weiter zur Evidenzsuche
    """
    if state.get("error"):
        return "error"

    decomposition = state.get("decomposition")
    if not decomposition or not decomposition.sub_claims:
        return "error"

    return "continue"


def should_continue_after_evidence(state: GraphState) -> str:
    """Entscheidet nach der Evidenzsuche, ob weitergemacht werden soll."""
    if state.get("error"):
        return "error"
    return "continue"


def should_continue_after_evaluation(state: GraphState) -> str:
    """Entscheidet nach der Bewertung, ob weitergemacht werden soll."""
    if state.get("error"):
        return "error"
    if not state.get("sub_verdicts"):
        return "error"
    return "continue"


# ---------------------------------------------------------------------------
# Graph bauen
# ---------------------------------------------------------------------------

def build_fact_check_graph() -> StateGraph:
    """
    Baut den LangGraph-Workflow für den Faktencheck.
    
    Ablauf:
    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │  Decompose   │───▶│  Retrieve        │───▶│  Evaluate        │───▶│  Synthesize      │
    │  Claim       │    │  Evidence         │    │  Evidence        │    │  Verdict         │
    └──────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
          │                    │                       │                       │
          ▼                    ▼                       ▼                       ▼
        [error]             [error]                 [error]                  [END]
    """

    # Graph erstellen
    workflow = StateGraph(GraphState)

    # Nodes hinzufügen
    workflow.add_node("decompose", decompose_claim)
    workflow.add_node("retrieve", retrieve_evidence)
    workflow.add_node("evaluate", evaluate_evidence)
    workflow.add_node("synthesize", synthesize_verdict)

    # Startpunkt
    workflow.set_entry_point("decompose")

    # Conditional Edges (Routing)
    workflow.add_conditional_edges(
        "decompose",
        should_continue_after_decomposition,
        {
            "continue": "retrieve",
            "error": END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        should_continue_after_evidence,
        {
            "continue": "evaluate",
            "error": END,
        },
    )

    workflow.add_conditional_edges(
        "evaluate",
        should_continue_after_evaluation,
        {
            "continue": "synthesize",
            "error": END,
        },
    )

    # Synthesize → Ende
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience: Graph einmal kompilieren
# ---------------------------------------------------------------------------

def run_fact_check(claim: str) -> GraphState:
    """
    Führt den kompletten Faktencheck für eine Behauptung durch.
    
    Args:
        claim: Die zu überprüfende Behauptung
    
    Returns:
        Der finale GraphState mit allen Ergebnissen
    """
    graph = build_fact_check_graph()

    initial_state: GraphState = {
        "claim": claim,
        "decomposition": None,
        "search_results": {},
        "sub_verdicts": [],
        "final_result": None,
        "error": None,
    }

    # Graph ausführen
    final_state = graph.invoke(initial_state)

    return final_state
