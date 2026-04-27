"""
LangGraph StateGraph definition for the Clinician Chatbot.
Defines the graph topology: router -> (stage_executor | summary) -> END.
Each invoke() call processes one conversation turn.
"""

import logging

from langgraph.graph import StateGraph, END

from nodes import router_node, stage_executor_node, summary_node

logger = logging.getLogger(__name__)

def _route_after_router(state: dict) -> str:
    """
    After the router picks a stage, decide which executor node to run.
    - 'generate_summary' -> summary node (produces structured clinical summary)
    - everything else    -> stage_executor (normal conversation)
    """
    trace = state.get("trace", {})
    current_stage = trace.get("current_stage", "")
    if current_stage == "generate_summary":
        return "summary"
    return "stage_executor"


def build_graph():
    """
    Constructs and compiles the clinician chatbot graph.

    Flow per invocation:
        START -> router -> (stage_executor | summary) -> END

    The outer conversation loop in main.py calls graph.invoke() once per turn.
    """

    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("stage_executor", stage_executor_node)
    graph.add_node("summary", summary_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge after router
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "stage_executor": "stage_executor",
            "summary": "summary",
        }
    )

    # Both executors lead to END (one turn per invoke)
    graph.add_edge("stage_executor", END)
    graph.add_edge("summary", END)

    return graph.compile()


clinician_graph = build_graph()
