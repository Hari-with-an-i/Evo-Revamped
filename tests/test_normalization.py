"""Tests for shared normalization helpers in src/workers/_normalization.py"""
import pytest
from datetime import datetime, timezone

from src.schemas import Article
from src.workers._normalization import (
    normalize_commoncrawl,
    normalize_gdelt,
    normalize_tavily,
    deduplicate,
    relevance_filter,
    classify_outlet,
)


def _make_article(url: str, title: str = "T") -> Article:
    return Article(
        title=title,
        body="Body text.",
        source_name="Source",
        source_url=url,
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        outlet_type="unknown",
        query_used="q",
    )


# ---------------------------------------------------------------------------
# classify_outlet
# ---------------------------------------------------------------------------

def test_classify_mainstream():
    assert classify_outlet("https://bbc.com/news/123", "BBC News") == "mainstream"


def test_classify_state_media():
    assert classify_outlet("https://rt.com/story", "RT") == "state_media"


def test_classify_unknown_returns_independent():
    assert classify_outlet("https://someblog.io/post", "Some Blog") == "independent"


# ---------------------------------------------------------------------------
# normalize_commoncrawl
# ---------------------------------------------------------------------------

def _raw_commoncrawl(**overrides) -> dict:
    base = {
        "_tool": "commoncrawl", "_query": "test",
        "title": "CC Article", "body": "Article body from Common Crawl.",
        "url": "https://example.com/article/1",
        "published_at": "2024-03-01T12:00:00Z",
        "source_name": "example.com",
    }
    base.update(overrides)
    return base


def test_normalize_commoncrawl_valid():
    a = normalize_commoncrawl(_raw_commoncrawl())
    assert isinstance(a, Article)
    assert a.body == "Article body from Common Crawl."
    assert a.source_name == "example.com"
    assert a.query_used == "test"


def test_normalize_commoncrawl_missing_url_returns_none():
    assert normalize_commoncrawl({"_tool": "commoncrawl", "url": ""}) is None


def test_normalize_commoncrawl_error_returns_none():
    assert normalize_commoncrawl({"_tool": "commoncrawl", "_error": "quota exceeded", "url": ""}) is None


# ---------------------------------------------------------------------------
# normalize_gdelt
# ---------------------------------------------------------------------------

def test_normalize_gdelt_valid():
    raw = {"_tool": "gdelt", "_query": "test", "title": "GDELT Article",
           "url": "https://gdelt-source.com/story", "domain": "gdelt-source.com",
           "seendate": "2024-03-01T08:00:00Z"}
    a = normalize_gdelt(raw)
    assert isinstance(a, Article)
    assert a.source_name == "gdelt-source.com"


# ---------------------------------------------------------------------------
# normalize_tavily
# ---------------------------------------------------------------------------

def test_normalize_tavily_valid():
    raw = {"_tool": "tavily", "_query": "test", "title": "Tavily Result",
           "url": "https://example.org/news", "raw_content": "Full content.",
           "source": "example.org"}
    a = normalize_tavily(raw)
    assert isinstance(a, Article)
    assert a.body == "Full content."


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_removes_same_url():
    articles = [_make_article("https://a.com"), _make_article("https://a.com"), _make_article("https://b.com")]
    assert len(deduplicate(articles)) == 2


def test_deduplicate_preserves_order():
    urls = ["https://a.com", "https://b.com", "https://c.com"]
    result = deduplicate([_make_article(u) for u in urls])
    assert [a.source_url for a in result] == urls


# ---------------------------------------------------------------------------
# relevance_filter
# ---------------------------------------------------------------------------

def test_relevance_filter_returns_top_n():
    articles = [_make_article(f"https://example.com/{i}", f"Article {i}") for i in range(50)]
    filtered = relevance_filter(articles, "vaccines COVID-19", top_n=10)
    assert len(filtered) == 10


def test_relevance_filter_passthrough_when_small():
    articles = [_make_article(f"https://example.com/{i}") for i in range(5)]
    assert len(relevance_filter(articles, "claim", top_n=10)) == 5
