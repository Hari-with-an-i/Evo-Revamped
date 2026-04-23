"""
Shared article normalization helpers.

Extracted from news_retrieval.py so they can be reused across all retrieval workers
without circular imports. No LLM calls — pure data transformation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.schemas import Article
from src.utils.html_utils import strip_html

_OUTLET_KEYWORDS: dict[str, list[str]] = {
    "state_media": ["xinhua", "rt.com", "tass", "sputnik", "cgtn", "presstv", "almayadeen"],
    "mainstream": [
        "bbc", "reuters", "apnews", "nytimes", "theguardian", "washingtonpost",
        "cnn", "nbcnews", "abcnews", "cbsnews", "npr", "wsj", "bloomberg",
    ],
}


def classify_outlet(url: str, name: str) -> str:
    combined = (url + name).lower()
    for outlet_type, keywords in _OUTLET_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return outlet_type
    return "independent"


_DATE_FORMATS = [
    "%Y%m%d%H%M%S",        # GDELT: 20200415120000
    "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 UTC
    "%Y-%m-%dT%H:%M:%S%z", # ISO 8601 with tz
    "%Y-%m-%d",             # date only
    "%d %b %Y",             # 15 Apr 2020
    "%B %d, %Y",            # April 15, 2020
]


def parse_date(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        raw = raw.strip()
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                continue
    return None


def normalize_commoncrawl(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    name = raw.get("source_name", url.split("/")[2] if "/" in url else url)
    return Article(
        title=raw.get("title", ""),
        body=strip_html(raw.get("body", "")),
        source_name=name,
        source_url=url,
        published_at=parse_date(raw.get("published_at")),
        outlet_type=classify_outlet(url, name),
        query_used=raw.get("_query", ""),
    )


def normalize_gdelt(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    domain = raw.get("domain", "")
    title = raw.get("title", "")
    return Article(
        title=title,
        body=raw.get("seendate", "") + " " + title,
        source_name=domain,
        source_url=url,
        published_at=parse_date(raw.get("seendate")),
        outlet_type=classify_outlet(url, domain),
        query_used=raw.get("_query", ""),
    )


_TAVILY_BODY_MAX = 1200  # max chars any downstream LLM ever reads from a Tavily body


def normalize_tavily(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    name = raw.get("source", url.split("/")[2] if "/" in url else url)
    body = strip_html((raw.get("raw_content") or raw.get("content") or "")[:_TAVILY_BODY_MAX])
    return Article(
        title=raw.get("title", ""),
        body=body,
        source_name=name,
        source_url=url,
        published_at=parse_date(raw.get("published_date")),
        outlet_type=classify_outlet(url, name),
        query_used=raw.get("_query", ""),
    )


def normalize_batch(raw_articles: list[dict]) -> list[Article]:
    _normalizers = {
        "commoncrawl": normalize_commoncrawl,
        "gdelt": normalize_gdelt,
        "tavily": normalize_tavily,
    }
    articles: list[Article] = []
    for raw in raw_articles:
        tool_name = raw.get("_tool", "")
        normalizer = _normalizers.get(tool_name)
        if normalizer is None:
            continue
        try:
            article = normalizer(raw)
            if article and article.title:
                articles.append(article)
        except Exception:
            continue
    return articles


def deduplicate(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        if a.id not in seen:
            seen.add(a.id)
            unique.append(a)
    return unique


def relevance_filter(articles: list[Article], claim: str, top_n: int = 30) -> list[Article]:
    if len(articles) <= top_n:
        return articles
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        corpus = [claim] + [f"{a.title} {a.body[:300]}" for a in articles]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [articles[i] for i in top_indices]
    except ImportError:
        return sorted(articles, key=lambda a: a.published_at, reverse=True)[:top_n]
