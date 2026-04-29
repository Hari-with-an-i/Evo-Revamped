"""
GdeltCommonCrawlTargetedWorker — Phase 2

Executes targeted GDELT + Common Crawl (BigQuery) searches in parallel (asyncio.gather)
for each TargetedQuery assigned to tool="gdelt_commoncrawl".

GDELT:        No API key required; date-filtered news corpus, global coverage.
Common Crawl: Requires GCP_PROJECT_ID + credentials; broader web archive coverage.
"""
from __future__ import annotations

import asyncio

from src.logger import get_logger
from src.state import AgentState
from src.tools.news_tools import gdelt_search, common_crawl_search
from src.workers._normalization import normalize_gdelt, normalize_commoncrawl, deduplicate

log = get_logger(__name__)

MAX_PER_QUERY = 15


async def _fetch_targeted(queries: list[dict]) -> list[dict]:
    loop = asyncio.get_event_loop()

    async def call(fn, query: str, **kwargs) -> list[dict]:
        return await loop.run_in_executor(None, lambda: fn.invoke({"query": query, **kwargs}))

    tasks = []
    for q in queries:
        q_str = q.get("query", "")
        start = q.get("start_date", "")
        end = q.get("end_date", "")
        
        tasks.append(call(gdelt_search, q_str, timespan="90d", max_records=MAX_PER_QUERY, start_date=start, end_date=end))
        tasks.append(call(common_crawl_search, q_str, max_results=MAX_PER_QUERY, date_from=start))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    raw: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            log.error("fetch error", extra={"error": str(r)})
        elif isinstance(r, list):
            raw.extend(r)
    return raw


def gdelt_commoncrawl_targeted_node(state: AgentState) -> dict:
    """Run targeted GDELT + Common Crawl queries and append deduplicated results to state."""
    all_queries = state.get("targeted_queries", [])
    my_queries = [q for q in all_queries if q.get("tool") == "gdelt_commoncrawl"]

    if not my_queries:
        log.warning("no gdelt_commoncrawl queries", extra={"node": "gdelt_commoncrawl_targeted"})
        return {
            "retrieved_articles": [],
            "worker_outputs": ["[gdelt_commoncrawl_targeted]\nNo queries assigned — skipping."],
        }

    log.info("fetching gdelt+commoncrawl", extra={"query_count": len(my_queries)})
    try:
        raw = asyncio.run(_fetch_targeted(my_queries))
    except RuntimeError:
        import nest_asyncio  # type: ignore[import]
        nest_asyncio.apply()
        raw = asyncio.run(_fetch_targeted(my_queries))

    articles = deduplicate([
        a
        for item in raw
        for a in [
            normalize_gdelt(item) if item.get("_tool") == "gdelt"
            else normalize_commoncrawl(item)
        ]
        if a is not None and a.title
    ])

    log.info("gdelt_commoncrawl_targeted done", extra={"articles": len(articles)})
    return {
        "retrieved_articles": [a.model_dump(mode="json") for a in articles],
        "worker_outputs": [
            f"[gdelt_commoncrawl_targeted]\n"
            f"Queries ({len(my_queries)}): {' | '.join([q.get('query', '') for q in my_queries])}\n"
            f"Articles fetched: {len(articles)}"
        ],
    }
