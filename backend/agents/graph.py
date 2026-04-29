"""
Vikas.ai — LangGraph State Machine
Wires the four agent nodes into a conditional, stateful graph.

Flow:
  intake → (emergency?) ──yes──→ synthesis (disclaimer)
                          │
                          no
                          ↓
                      retrieval → reasoning → synthesis
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from backend.agents.nodes import (
    intake_node,
    reasoning_node,
    retrieval_node,
    synthesis_node,
)
from backend.agents.state import Severity

logger = logging.getLogger("vikas.agents.graph")


# ── Routing function ────────────────────────────────────────
def _route_after_intake(state: dict[str, Any]) -> str:
    """
    After intake, decide the next step:
      • EMERGENCY → skip retrieval/reasoning, go straight to synthesis
        (which will inject the hard-coded disclaimer).
      • Everything else → proceed to retrieval.
    """
    severity = state.get("severity", Severity.LOW.value)
    if severity in (Severity.EMERGENCY.value, Severity.EMERGENCY):
        logger.warning("Emergency detected — fast-tracking to synthesis with disclaimer")
        return "synthesis"
    return "retrieval"


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph reasoning pipeline.

    Returns a compiled graph that accepts an AgentState dict and
    returns the mutated state after traversing all nodes.
    """
    graph = StateGraph(dict)

    # ── Register nodes ──────────────────────────────────────
    graph.add_node("intake", intake_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("synthesis", synthesis_node)

    # ── Define edges ────────────────────────────────────────
    graph.set_entry_point("intake")

    # Conditional branch after intake
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {
            "retrieval": "retrieval",
            "synthesis": "synthesis",
        },
    )

    # Linear chain: retrieval → reasoning → synthesis → END
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "synthesis")
    graph.add_edge("synthesis", END)

    compiled = graph.compile()
    logger.info("LangGraph reasoning pipeline compiled successfully")
    return compiled


# Singleton compiled graph
reasoning_pipeline = build_graph()


async def run_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the full reasoning pipeline for a single conversational turn.

    Args:
        state: A dict matching the AgentState schema with at minimum
               `user_utterance` populated.

    Returns:
        The fully mutated state dict including `final_response_en`,
        `citations`, `reasoning_chain`, etc.
    """
    logger.info("Running pipeline for session=%s", state.get("session_id", "unknown"))
    result = await reasoning_pipeline.ainvoke(state)
    logger.info(
        "Pipeline complete — severity=%s, confidence=%.2f, citations=%d",
        result.get("severity", "?"),
        result.get("confidence", 0.0),
        len(result.get("citations", [])),
    )
    return result
