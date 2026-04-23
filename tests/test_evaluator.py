"""Tests for the Evaluator agent — mocks SBERT and NLI to avoid model downloads."""
import pytest
from unittest.mock import patch, MagicMock

from src.workers.evaluator import (
    _domain_root,
    _dedup_by_domain_root,
    _relevance_filter,
    evaluator_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(url: str, title: str = "Test", body: str = "Some content.", id: str = "") -> dict:
    return {
        "id": id or url.split("/")[-1],
        "title": title,
        "body": body,
        "source_name": url.split("/")[2],
        "source_url": url,
        "published_at": "2024-01-01T00:00:00+00:00",
        "outlet_type": "unknown",
        "query_used": "q",
    }


def _base_state(**overrides) -> dict:
    base = {
        "claim": "5G towers spread COVID-19.",
        "context_summary": {
            "who": "WHO", "what": "5G claim", "when": "2020",
            "competing_narratives": "",
            "claim_dimensions": "causal mechanism|source credibility",
        },
        "retrieved_articles": [],
        "anchor_articles": [],
        "worker_outputs": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Step 1 — domain-root deduplication
# ---------------------------------------------------------------------------

def test_domain_root_extraction():
    assert _domain_root("https://www.bbc.co.uk/news/article") == "co.uk"
    assert _domain_root("https://reuters.com/article/123") == "reuters.com"
    assert _domain_root("https://sub.nytimes.com/story") == "nytimes.com"


@patch("src.workers.evaluator.cosine_similarities")
def test_dedup_keeps_one_per_domain_root(mock_sims):
    mock_sims.return_value = [0.8, 0.6]  # first article scores higher
    articles = [
        _make_article("https://bbc.com/1", id="a1"),
        _make_article("https://bbc.com/2", id="a2"),   # same root
        _make_article("https://reuters.com/1", id="a3"),  # different root
    ]
    result = _dedup_by_domain_root(articles, "5G towers spread COVID-19.")
    roots = [_domain_root(a["source_url"]) for a in result]
    assert len(result) == 2
    assert len(set(roots)) == 2
    # a1 should be kept (higher score), a2 dropped
    ids = {a["id"] for a in result}
    assert "a1" in ids
    assert "a2" not in ids
    assert "a3" in ids


def test_dedup_different_domains_all_kept():
    articles = [
        _make_article("https://bbc.com/1"),
        _make_article("https://reuters.com/1"),
        _make_article("https://ap.org/1"),
    ]
    result = _dedup_by_domain_root(articles, "claim")
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Step 2 — SBERT relevance filter
# ---------------------------------------------------------------------------

@patch("src.workers.evaluator.cosine_similarities")
def test_relevance_filter_drops_low_cosine(mock_sims):
    mock_sims.return_value = [0.8, 0.3, 0.55]  # second dropped, third peripheral
    articles = [
        _make_article("https://a.com/1"),
        _make_article("https://b.com/1"),  # will be dropped
        _make_article("https://c.com/1"),  # will be peripheral
    ]
    relevant, peripheral = _relevance_filter(articles, "claim")
    assert len(relevant) == 1
    assert len(peripheral) == 1
    assert relevant[0]["source_url"] == "https://a.com/1"
    assert peripheral[0].get("_peripheral") is True


@patch("src.workers.evaluator.cosine_similarities")
def test_relevance_filter_empty_input(mock_sims):
    mock_sims.return_value = []
    relevant, peripheral = _relevance_filter([], "claim")
    assert relevant == []
    assert peripheral == []


# ---------------------------------------------------------------------------
# evaluator_node — empty articles
# ---------------------------------------------------------------------------

def test_evaluator_empty_articles():
    result = evaluator_node(_base_state(retrieved_articles=[]))
    assert result["evaluation_ready"] is True
    assert result["perspective_clusters"] == []
    assert result["coverage_gaps"] == []
    assert "[evaluator]" in result["worker_outputs"][0]


# ---------------------------------------------------------------------------
# evaluator_node — full pipeline with mocks
# ---------------------------------------------------------------------------

@patch("src.workers.evaluator._detect_gaps")
@patch("src.workers.evaluator._build_clusters")
@patch("src.workers.evaluator._extract_claims")
@patch("src.workers.evaluator._relevance_filter")
@patch("src.workers.evaluator._dedup_by_domain_root")
def test_evaluator_node_returns_clusters(
    mock_dedup, mock_filter, mock_extract, mock_build, mock_gaps
):
    from src.schemas.perspective_cluster import PerspectiveCluster

    articles = [_make_article(f"https://site{i}.com/1") for i in range(3)]
    mock_dedup.return_value = articles
    mock_filter.return_value = (articles, [])
    mock_extract.return_value = [["claim A", "claim B"]] * 3
    mock_cluster = PerspectiveCluster(
        label="5G is harmful according to some sources.",
        key_claims=["claim A", "claim B"],
        article_ids=["1", "2", "3"],
        domain_roots=["site0.com", "site1.com", "site2.com"],
        corroboration_count=3,
        corroboration_level="partial",
    )
    mock_build.return_value = [mock_cluster]
    mock_gaps.return_value = []

    result = evaluator_node(_base_state(retrieved_articles=articles))

    assert len(result["perspective_clusters"]) == 1
    assert result["perspective_clusters"][0]["corroboration_level"] == "partial"
    assert result["evaluation_ready"] is True
    assert result["coverage_gaps"] == []
    assert "clusters: 1" in result["worker_outputs"][0].lower()


# ---------------------------------------------------------------------------
# Step 4 — coverage gap detection (via evaluator_node)
# ---------------------------------------------------------------------------

@patch("src.workers.evaluator._detect_gaps")
@patch("src.workers.evaluator._build_clusters")
@patch("src.workers.evaluator._extract_claims")
@patch("src.workers.evaluator._relevance_filter")
@patch("src.workers.evaluator._dedup_by_domain_root")
def test_coverage_gap_sets_not_ready(
    mock_dedup, mock_filter, mock_extract, mock_build, mock_gaps
):
    from src.schemas.perspective_cluster import PerspectiveCluster

    articles = [_make_article(f"https://site{i}.com/1") for i in range(2)]
    mock_dedup.return_value = articles
    mock_filter.return_value = (articles, [])
    mock_extract.return_value = [["claim A", "claim B"]] * 2
    mock_cluster = PerspectiveCluster(
        label="Some viewpoint.",
        key_claims=["claim A"],
        article_ids=["1"],
        domain_roots=["site0.com"],
        corroboration_count=1,
        corroboration_level="unverified",
    )
    mock_build.return_value = [mock_cluster]
    mock_gaps.return_value = ["source credibility"]  # gap remains

    result = evaluator_node(_base_state(retrieved_articles=articles))

    assert result["evaluation_ready"] is False
    assert "source credibility" in result["coverage_gaps"]
    assert "Gaps: ['source credibility']" in result["worker_outputs"][0]
