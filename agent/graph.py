"""
FactAgent – LangGraph Workflow (v2: Async)
==========================================
Hier wird der Agentic Workflow als Graph definiert.

Update v2:
- Nodes sind jetzt async → Graph nutzt ainvoke()
- run_fact_check() ist async und wird vom Eval-Script via asyncio aufgerufen
- Die Chainlit-App (app.py) ruft Nodes direkt auf für bessere Step-Kontrolle
"""

import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from agent.models import (
    ClaimDecomposition,
    ClaimType,
    FactCheckResult,
    HumanFeedback,
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
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """State, der durch den Workflow fliesst."""
    claim: str
    decomposition: Optional[ClaimDecomposition]
    search_results: dict
    sub_verdicts: list[SubClaimVerdict]
    human_feedback: Optional[HumanFeedback]
    final_result: Optional[FactCheckResult]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Routing-Logik
# ---------------------------------------------------------------------------

def should_continue_after_decomposition(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    decomposition = state.get("decomposition")
    if not decomposition or not decomposition.sub_claims:
        return "error"
    return "continue"


def should_continue_after_evidence(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


def should_continue_after_evaluation(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("sub_verdicts"):
        return "error"
    return "continue"


# ---------------------------------------------------------------------------
# Async Node Wrappers (ohne on_token für Graph-Nutzung)
# ---------------------------------------------------------------------------

async def _decompose(state: dict) -> dict:
    return await decompose_claim(state)

async def _retrieve(state: dict) -> dict:
    return await retrieve_evidence(state)

async def _evaluate(state: dict) -> dict:
    return await evaluate_evidence(state)

async def _synthesize(state: dict) -> dict:
    return await synthesize_verdict(state)


# ---------------------------------------------------------------------------
# Graph bauen
# ---------------------------------------------------------------------------

def build_fact_check_graph() -> StateGraph:
    """Baut den LangGraph-Workflow für den Faktencheck."""

    workflow = StateGraph(GraphState)

    # Nodes hinzufügen (async)
    workflow.add_node("decompose", _decompose)
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("evaluate", _evaluate)
    workflow.add_node("synthesize", _synthesize)

    workflow.set_entry_point("decompose")

    workflow.add_conditional_edges(
        "decompose",
        should_continue_after_decomposition,
        {"continue": "retrieve", "error": END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        should_continue_after_evidence,
        {"continue": "evaluate", "error": END},
    )
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_after_evaluation,
        {"continue": "synthesize", "error": END},
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


async def run_fact_check(claim: str) -> GraphState:
    """
    Führt den kompletten Faktencheck async durch.
    
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
        "human_feedback": None,
        "final_result": None,
        "error": None,
    }

    # Async Graph ausführen
    final_state = await graph.ainvoke(initial_state)
    return final_state
