"""
News retrieval tools — GDELT, Tavily, Common Crawl (BigQuery).

Each tool fetches articles from a single source and returns a list of raw dicts
tagged with `_tool` and `_query`. Normalization into Article schema happens in
the worker, not here.

Required env vars:
    TAVILY_API_KEY           — https://tavily.com
    GCP_PROJECT_ID           — Google Cloud project that has BigQuery billing enabled
    GOOGLE_APPLICATION_CREDENTIALS — path to service-account JSON (or use ADC)
    COMMON_CRAWL_BQ_TABLE    — fully-qualified BigQuery table for CC-News queries
                               default: commoncrawl.cc_news.articles
    GDELT_BASE_URL           — override GDELT endpoint (default: public GDELT 2.0 API)
"""
from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

_GDELT_BASE = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")


@tool
def gdelt_search(query: str, timespan: str = "30d", max_records: int = 20) -> list[dict]:
    """
    Search the GDELT Document 2.0 API (no API key required).

    Args:
        query:       Search string.
        timespan:    Time window — e.g. '7d', '24h', '30d' (max ~3 months).
        max_records: Max articles to return (GDELT hard cap: 250).

    Returns:
        List of raw article dicts from GDELT.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": min(max_records, 250),
        "timespan": timespan,
        "sort": "HybridRel",
        "format": "json",
    }
    try:
        resp = httpx.get(_GDELT_BASE, params=params, timeout=20)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        for a in articles:
            a["_tool"] = "gdelt"
            a["_query"] = query
        return articles
    except Exception as exc:
        return [{"_error": str(exc), "_tool": "gdelt", "_query": query}]


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------

@tool
def tavily_search(query: str, max_results: int = 10) -> list[dict]:
    """
    Web search via Tavily — full-text, recent results.

    Args:
        query:       Search string.
        max_results: Max results to return (Tavily cap: 20).

    Returns:
        List of raw result dicts from Tavily.
    """
    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        return [{"_error": "tavily-python not installed", "_tool": "tavily", "_query": query}]

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=min(max_results, 20),
            include_raw_content=True,
        )
        results = response.get("results", [])
        for r in results:
            r["_tool"] = "tavily"
            r["_query"] = query
        return results
    except Exception as exc:
        return [{"_error": str(exc), "_tool": "tavily", "_query": query}]


# ---------------------------------------------------------------------------
# Common Crawl — Google BigQuery
# ---------------------------------------------------------------------------

# Default table is the CC-News public dataset on BigQuery.
# Override via COMMON_CRAWL_BQ_TABLE env var if you have a private mirror or
# a different CC index table (e.g. commoncrawl.cc_index.YYYY_WW).
_CC_BQ_TABLE = os.getenv("COMMON_CRAWL_BQ_TABLE", "commoncrawl.cc_news.articles")
_GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")


@tool
def common_crawl_search(query: str, max_results: int = 15, date_from: str = "") -> list[dict]:
    """
    Search Common Crawl news articles via Google BigQuery.

    Requires:
        - GCP_PROJECT_ID env var (billing-enabled GCP project)
        - Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS

    Args:
        query:       Keyword/phrase to search in article titles and content.
        max_results: Max rows to return.
        date_from:   Optional ISO date lower bound (YYYY-MM-DD). Filters publish_date.

    Returns:
        List of raw article dicts from Common Crawl.
    """
    if not _GCP_PROJECT:
        return [{"_error": "GCP_PROJECT_ID not set", "_tool": "commoncrawl", "_query": query}]

    try:
        from google.cloud import bigquery  # type: ignore[import]
    except ImportError:
        return [{"_error": "google-cloud-bigquery not installed", "_tool": "commoncrawl", "_query": query}]

    try:
        client = bigquery.Client(project=_GCP_PROJECT)

        date_filter = ""
        if date_from:
            date_filter = f"AND DATE(publish_date) >= '{date_from}'"

        sql = f"""
            SELECT
                url,
                COALESCE(title, '') AS title,
                COALESCE(SUBSTR(content, 1, 2000), '') AS body,
                url_host_name AS source_name,
                COALESCE(CAST(publish_date AS STRING), CAST(crawl_date AS STRING)) AS published_at
            FROM `{_CC_BQ_TABLE}`
            WHERE (
                LOWER(title)   LIKE @kw
                OR LOWER(content) LIKE @kw
            )
            {date_filter}
            ORDER BY publish_date DESC
            LIMIT @max_results
        """

        kw_param = f"%{query.lower()}%"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("kw", "STRING", kw_param),
                bigquery.ScalarQueryParameter("max_results", "INT64", max_results),
            ]
        )

        rows = client.query(sql, job_config=job_config).result()
        results = []
        for row in rows:
            d = dict(row)
            d["_tool"] = "commoncrawl"
            d["_query"] = query
            results.append(d)
        return results

    except Exception as exc:
        return [{"_error": str(exc), "_tool": "commoncrawl", "_query": query}]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

news_tools = [gdelt_search, tavily_search, common_crawl_search]
