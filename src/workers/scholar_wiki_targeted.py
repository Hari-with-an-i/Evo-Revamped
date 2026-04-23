"""
ScholarWikiTargetedWorker — Phase 2

Fetches high-credibility anchor articles from Wikipedia for each TargetedQuery
assigned to tool="scholar_wiki". No API key required.

wikipedia package docs: https://pypi.org/project/wikipedia/
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.logger import get_logger
from src.schemas import Article
from src.state import AgentState
from src.workers._normalization import deduplicate

log = get_logger(__name__)


def _fetch_wikipedia(query: str) -> list[Article]:
    try:
        import wikipedia  # type: ignore[import]
        wikipedia.set_lang("en")
        search_titles = wikipedia.search(query, results=3)
        articles: list[Article] = []
        for title in search_titles[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                articles.append(Article(
                    title=page.title,
                    body=page.summary[:2000],
                    source_name="Wikipedia",
                    source_url=page.url,
                    published_at=datetime.now(timezone.utc),
                    outlet_type="mainstream",
                    query_used=query,
                ))
            except Exception as e:
                log.debug("wikipedia page error", extra={"title": title, "error": str(e)})
                continue
        return articles
    except ImportError:
        log.warning("wikipedia package not installed")
        return []


def scholar_wiki_targeted_node(state: AgentState) -> dict:
    """Fetch Wikipedia anchor pages for scholar_wiki targeted queries."""
    all_queries = state.get("targeted_queries", [])
    my_queries = [q["query"] for q in all_queries if q.get("tool") == "scholar_wiki"]

    if not my_queries:
        log.warning("no scholar_wiki queries", extra={"node": "scholar_wiki_targeted"})
        return {
            "retrieved_articles": [],
            "worker_outputs": ["[scholar_wiki_targeted]\nNo queries assigned — skipping."],
        }

    articles: list[Article] = []
    for q in my_queries:
        articles.extend(_fetch_wikipedia(q))

    articles = deduplicate(articles)

    log.info("scholar_wiki_targeted done", extra={"articles": len(articles)})
    return {
        "anchor_articles": [a.model_dump(mode="json") for a in articles],
        "worker_outputs": [
            f"[scholar_wiki_targeted]\n"
            f"Queries ({len(my_queries)}): {' | '.join(my_queries)}\n"
            f"Anchor pages fetched: {len(articles)} (Wikipedia — anchoring only, not cited in report)"
        ],
    }
