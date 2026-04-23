"""Tests for Phase 1 workers: broad_context_fetch and context_builder."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.schemas import Article, ContextSummary
from src.workers.broad_context_fetch import broad_context_fetch_node
from src.workers.context_builder import context_builder_node


def _base_state(**overrides) -> dict:
    base = {
        "claim": "5G towers spread COVID-19.",
        "entities": ["5G", "COVID-19"],
        "timeframe": None,
        "retrieved_articles": [],
        "worker_outputs": [],
    }
    base.update(overrides)
    return base


def _make_tavily_raw(url: str, title: str, query: str = "test") -> dict:
    return {
        "_tool": "tavily", "_query": query,
        "title": title, "content": "Some content here.",
        "url": url, "source": url.split("/")[2],
    }


# ---------------------------------------------------------------------------
# broad_context_fetch
# ---------------------------------------------------------------------------

@patch("src.workers.broad_context_fetch.tavily_search")
def test_broad_context_fetch_returns_articles(mock_tavily):
    mock_tavily.invoke.return_value = [
        _make_tavily_raw("https://bbc.com/1", "BBC Story", "5G towers spread COVID-19."),
        _make_tavily_raw("https://reuters.com/2", "Reuters Story", "5G towers spread COVID-19."),
    ]

    result = broad_context_fetch_node(_base_state())

    assert len(result["retrieved_articles"]) == 2
    assert "[broad_context_fetch]" in result["worker_outputs"][0]


@patch("src.workers.broad_context_fetch.tavily_search")
def test_broad_context_fetch_deduplicates(mock_tavily):
    raw = _make_tavily_raw("https://same.com/article", "Same Article")
    mock_tavily.invoke.return_value = [raw, raw]

    result = broad_context_fetch_node(_base_state())
    assert len(result["retrieved_articles"]) == 1


def test_broad_context_fetch_empty_claim():
    result = broad_context_fetch_node(_base_state(claim=""))
    assert result["retrieved_articles"] == []
    assert "skipping" in result["worker_outputs"][0].lower()


# ---------------------------------------------------------------------------
# context_builder
# ---------------------------------------------------------------------------

@patch("src.workers.context_builder._get_llm")
def test_context_builder_produces_summary(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = ContextSummary(
        who="WHO", what="5G spread claim", when="2020",
        competing_narratives="5G is safe",
        claim_dimensions="causal mechanism|timeline",
    )
    mock_get_llm.return_value = mock_llm

    articles = [Article(
        title="Test", body="Body.", source_name="S",
        source_url="https://example.com/1",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        outlet_type="unknown", query_used="q",
    )]
    state = _base_state(retrieved_articles=[a.model_dump(mode="json") for a in articles])

    result = context_builder_node(state)

    assert result["context_summary"]["who"] == "WHO"
    assert "causal mechanism" in result["context_summary"]["claim_dimensions"].split("|")
    assert "[context_builder]" in result["worker_outputs"][0]


def test_context_builder_no_articles_produces_minimal_summary():
    result = context_builder_node(_base_state(retrieved_articles=[]))
    assert result["context_summary"]["what"] == "5G towers spread COVID-19."
    assert "minimal" in result["worker_outputs"][0].lower()
