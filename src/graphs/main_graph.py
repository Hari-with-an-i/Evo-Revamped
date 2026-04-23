from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.orchestrator import orchestrator_node, route_from_orchestrator
from src.workers import (
    broad_context_fetch_node,
    context_builder_node,
    tavily_targeted_node,
    gdelt_commoncrawl_targeted_node,
    scholar_wiki_targeted_node,
    evaluator_node,
    analyst_node,
    researcher_node,
    writer_node,
)
from src.config import config

# ---------------------------------------------------------------------------
# Worker registry — only workers the orchestrator routes TO directly.
# context_builder and analyst are wired via direct edges and excluded here.
# ---------------------------------------------------------------------------
ORCHESTRATOR_ROUTED_WORKERS: dict = {
    "broad_context_fetch":        broad_context_fetch_node,
    "tavily_targeted":            tavily_targeted_node,
    "gdelt_commoncrawl_targeted": gdelt_commoncrawl_targeted_node,
    "scholar_wiki_targeted":      scholar_wiki_targeted_node,
    "evaluator":                  evaluator_node,
    "researcher":                 researcher_node,
    "writer":                     writer_node,
}

# Phase 2 + pipeline workers that loop back to orchestrator after running.
# evaluator is excluded — it feeds directly into analyst, which returns to orchestrator.
_RETURN_TO_ORCHESTRATOR = [
    "tavily_targeted",
    "gdelt_commoncrawl_targeted",
    "scholar_wiki_targeted",
    "researcher",
    "writer",
]


def build_graph():
    """
    Orchestrator-worker graph for misinformation analysis.

    Execution flow:
        START
          → orchestrator (MODE A: parse input)
          → broad_context_fetch → context_builder   [Phase 1 direct chain]
          → orchestrator (MODE B: plan targeted queries)
          → tavily_targeted                          [Phase 2 — sequential MVP]
          → orchestrator (MODE C: evaluate corpus)
          → newsapi_gdelt_targeted  ┐
          → orchestrator            ├ up to MAX_RETRIEVAL_LOOPS times
          → scholar_wiki_targeted   ┘
          → orchestrator (corpus sufficient → researcher)
          → orchestrator (→ writer)
          → orchestrator (→ FINISH → END)

    Phase 2 parallel upgrade path:
        Replace sequential add_edge calls with Send-based fan-out from
        route_from_orchestrator. No state or orchestrator changes needed.
    """
    graph = StateGraph(AgentState)

    # --- Register all nodes --------------------------------------------------
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("context_builder", context_builder_node)
    for name, node_fn in ORCHESTRATOR_ROUTED_WORKERS.items():
        graph.add_node(name, node_fn)

    # --- Edges ---------------------------------------------------------------

    # Entry point
    graph.add_edge(START, "orchestrator")

    graph.add_node("analyst", analyst_node)

    # Phase 1: unconditional chain — orchestrator never routes to context_builder
    graph.add_edge("broad_context_fetch", "context_builder")
    graph.add_edge("context_builder", "orchestrator")

    # Orchestrator conditional routing
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {**{name: name for name in ORCHESTRATOR_ROUTED_WORKERS}, "FINISH": END},
    )

    # Phase 2 + pipeline workers return to orchestrator
    for name in _RETURN_TO_ORCHESTRATOR:
        graph.add_edge(name, "orchestrator")

    # Analyst runs directly after evaluator (not orchestrator-routed),
    # then returns to orchestrator which checks analysis_complete before routing to researcher.
    graph.add_edge("evaluator", "analyst")
    graph.add_edge("analyst", "orchestrator")

    return graph.compile()
