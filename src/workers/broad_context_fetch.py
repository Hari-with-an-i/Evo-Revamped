"""
BroadContextFetchWorker — Phase 1, Step 1

Fast Tavily fetch using 1–2 wide queries derived directly from the claim and entities.
No LLM call — keeps Phase 1 cheap and fast.
Writes raw articles into retrieved_articles for context_builder to analyse.
"""
from src.logger import get_logger
from src.state import AgentState
from src.tools.news_tools import tavily_search
from src.workers._normalization import normalize_tavily, deduplicate

log = get_logger(__name__)

MAX_RESULTS_PER_QUERY = 8


def broad_context_fetch_node(state: AgentState) -> dict:
    """Fetch broad context articles via Tavily using the claim and entity cluster."""
    claim = state.get("claim", "")
    entities = state.get("entities", [])

    if not claim:
        log.warning("skipping — empty claim", extra={"node": "broad_context_fetch"})
        return {
            "retrieved_articles": [],
            "worker_outputs": ["[broad_context_fetch]\nNo claim in state — skipping."],
        }

    queries = [claim]
    if entities:
        queries.append(" ".join(entities[:3]))

    raw_results: list[dict] = []
    for q in queries[:2]:
        log.debug("tavily query", extra={"query": q})
        results = tavily_search.invoke({"query": q, "max_results": MAX_RESULTS_PER_QUERY})
        if isinstance(results, list):
            raw_results.extend(results)

    articles = deduplicate([
        a for raw in raw_results
        if (a := normalize_tavily(raw)) is not None and a.title
    ])

    log.info("broad_context_fetch done", extra={"queries": len(queries[:2]), "articles": len(articles)})
    return {
        "retrieved_articles": [a.model_dump(mode="json") for a in articles],
        "worker_outputs": [
            f"[broad_context_fetch]\n"
            f"Queries ({len(queries[:2])}): {' | '.join(queries[:2])}\n"
            f"Articles fetched: {len(articles)}"
        ],
    }
