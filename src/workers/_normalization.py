"""
Shared article normalization helpers.

Extracted from news_retrieval.py so they can be reused across all retrieval workers
without circular imports. No LLM calls — pure data transformation.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.schemas import Article
from src.utils.html_utils import strip_html

_MIN_BODY_CHARS = 100   # minimum chars for substantive article body
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

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


def extract_substantive_body(cleaned: str, max_chars: int = 1200) -> str:
    """
    From already-cleaned text, extract the first run of prose >= _MIN_BODY_CHARS.
    Returns empty string if no substantive content found (login wall, redirect page).
    """
    if not cleaned:
        return ""
    segments = _SENT_SPLIT.split(cleaned)
    accumulated: list[str] = []
    total = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        accumulated.append(seg)
        total += len(seg)
        if total >= _MIN_BODY_CHARS:
            break
    return " ".join(accumulated)[:max_chars]


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


_URL_DATE_NUMERIC = re.compile(r'/(\d{4})/(\d{2})/(\d{2})/')
_URL_DATE_SLUG = re.compile(
    r'/(\d{4})/(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/(\d{1,2})/',
    re.IGNORECASE,
)
_URL_YEAR_ONLY = re.compile(r'/(\d{4})/')

_MONTH_ABBR_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_FULL_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_BODY_DATE_PATTERNS = [
    # "17.04.2024" or "17-04-2024"
    (re.compile(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b'), "dmy"),
    # "April 17, 2024" or "April 2024"
    (re.compile(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'(?:\s+(\d{1,2}),?)?\s+(\d{4})\b', re.IGNORECASE,
    ), "mdy_long"),
    # "17 Apr 2024"
    (re.compile(
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        re.IGNORECASE,
    ), "dmy_abbr"),
    # "Published: 2024-02-17" or "Updated: 2024-02-17"
    (re.compile(r'(?:Published|Updated|Date)[:\s]+(\d{4})-(\d{2})-(\d{2})', re.IGNORECASE), "ymd"),
]


def _infer_date_from_url(url: str) -> datetime | None:
    """Extract publication date embedded in news URL path segments."""
    if not url:
        return None
    m = _URL_DATE_NUMERIC.search(url)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                return datetime(y, mo, d, tzinfo=timezone.utc)
            except ValueError:
                pass
    m = _URL_DATE_SLUG.search(url)
    if m:
        y = int(m.group(1))
        mo = _MONTH_ABBR_MAP.get(m.group(2).lower(), 0)
        d = int(m.group(3))
        if 1990 <= y <= 2030 and mo and 1 <= d <= 31:
            try:
                return datetime(y, mo, d, tzinfo=timezone.utc)
            except ValueError:
                pass
    m = _URL_YEAR_ONLY.search(url)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2030:
            return datetime(y, 7, 1, tzinfo=timezone.utc)
    return None


def _infer_date_from_body(body: str, scan_chars: int = 500) -> datetime | None:
    """Scan the first scan_chars of cleaned body for a publication date pattern."""
    snippet = body[:scan_chars] if body else ""
    if not snippet:
        return None
    for pattern, fmt in _BODY_DATE_PATTERNS:
        m = pattern.search(snippet)
        if not m:
            continue
        try:
            if fmt == "dmy":
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1990 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return datetime(y, mo, d, tzinfo=timezone.utc)
            elif fmt == "mdy_long":
                mo = _FULL_MONTH_MAP.get(m.group(1).lower(), 0)
                day_str = m.group(2)
                d = int(day_str) if day_str else 1
                y = int(m.group(3))
                if 1990 <= y <= 2030 and mo:
                    return datetime(y, mo, d, tzinfo=timezone.utc)
            elif fmt == "dmy_abbr":
                d = int(m.group(1))
                mo = _MONTH_ABBR_MAP.get(m.group(2).lower(), 0)
                y = int(m.group(3))
                if 1990 <= y <= 2030 and mo:
                    return datetime(y, mo, d, tzinfo=timezone.utc)
            elif fmt == "ymd":
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1990 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return datetime(y, mo, d, tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            continue
    return None


def _resolve_date(raw_date: Any, url: str, body: str) -> datetime | None:
    """Priority chain: explicit field → URL path → body text."""
    return parse_date(raw_date) or _infer_date_from_url(url) or _infer_date_from_body(body)


def normalize_commoncrawl(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    name = raw.get("source_name", url.split("/")[2] if "/" in url else url)
    cleaned = strip_html(raw.get("body", ""))
    body = extract_substantive_body(cleaned)
    if not body:
        body = strip_html(raw.get("title", ""))
    return Article(
        title=raw.get("title", ""),
        body=body,
        source_name=name,
        source_url=url,
        published_at=_resolve_date(raw.get("published_at"), url, body),
        outlet_type=classify_outlet(url, name),
        query_used=raw.get("_query", ""),
    )


def normalize_gdelt(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    domain = raw.get("domain", "")
    title = raw.get("title", "")
    seendate = raw.get("seendate", "")
    body = f"[{seendate}] {title}. Source: {domain}."
    return Article(
        title=title,
        body=body,
        source_name=domain,
        source_url=url,
        published_at=_resolve_date(seendate, url, ""),
        outlet_type=classify_outlet(url, domain),
        query_used=raw.get("_query", ""),
    )


_TAVILY_BODY_MAX = 1200  # max chars any downstream LLM ever reads from a Tavily body


def normalize_tavily(raw: dict) -> Article | None:
    if "_error" in raw or not raw.get("url"):
        return None
    url = raw.get("url", "")
    name = raw.get("source", url.split("/")[2] if "/" in url else url)
    # Clean from 4× the target budget so boilerplate doesn't eat the entire allowance
    raw_text = (raw.get("raw_content") or raw.get("content") or "")
    cleaned = strip_html(raw_text[:_TAVILY_BODY_MAX * 4])
    body = extract_substantive_body(cleaned, max_chars=_TAVILY_BODY_MAX)
    if not body:
        body = strip_html(raw.get("title", ""))
    return Article(
        title=raw.get("title", ""),
        body=body,
        source_name=name,
        source_url=url,
        published_at=_resolve_date(raw.get("published_date"), url, body),
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
                # Drop login-wall/redirect articles that have no substantive body.
                # GDELT articles are intentionally short (headline-only) — skip this filter for them.
                if tool_name != "gdelt" and len(article.body) < _MIN_BODY_CHARS:
                    continue
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


def deduplicate_dicts(articles: list[dict]) -> list[dict]:
    """Same dedup logic as deduplicate() but for pre-serialized article dicts."""
    seen: set[str] = set()
    unique: list[dict] = []
    for a in articles:
        aid = a.get("id", "")
        if aid not in seen:
            seen.add(aid)
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
