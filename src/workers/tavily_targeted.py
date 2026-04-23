"""
TavilyTargetedWorker — Phase 2

Executes targeted Tavily searches for each TargetedQuery assigned to tool="tavily".
Provides full-text, recent web results per claim dimension.
"""
from src.logger import get_logger
from src.state import AgentState
from src.tools.news_tools import tavily_search
from src.workers._normalization import normalize_tavily, deduplicate

log = get_logger(__name__)

MAX_PER_QUERY = 5


def tavily_targeted_node(state: AgentState) -> dict:
    """Run targeted Tavily queries from state and append results to retrieved_articles."""
    all_queries = state.get("targeted_queries", [])
    my_queries = [q for q in all_queries if q.get("tool") == "tavily"]

    if not my_queries:
        log.warning("no tavily queries assigned", extra={"node": "tavily_targeted"})
        return {
            "retrieved_articles": [],
            "worker_outputs": ["[tavily_targeted]\nNo targeted queries assigned — skipping."],
        }

    raw_results: list[dict] = []
    for q_dict in my_queries:
        query = q_dict["query"]
        log.debug("tavily targeted query", extra={"query": query})
        results = tavily_search.invoke({"query": query, "max_results": MAX_PER_QUERY})
        if isinstance(results, list):
            raw_results.extend(results)

    articles = deduplicate([
        a for raw in raw_results
        if (a := normalize_tavily(raw)) is not None and a.title
    ])

    query_strs = [q["query"] for q in my_queries]
    log.info("tavily_targeted done", extra={"queries": len(my_queries), "articles": len(articles)})
    return {
        "retrieved_articles": [a.model_dump(mode="json") for a in articles],
        "worker_outputs": [
            f"[tavily_targeted]\n"
            f"Queries ({len(my_queries)}): {' | '.join(query_strs)}\n"
            f"Articles fetched: {len(articles)}"
        ],
    }
