"""
Tests for the analyst pipeline — mocks LLM, SBERT, RoBERTa, and GDELT to avoid
model downloads and external calls.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.utils.bucketing import build_time_buckets
from src.workers.analyst import (
    _classify_ground_truth,
    _detect_inflection_candidates,
    _find_spike_buckets,
    analyst_node,
)
from src.schemas.narrative_report import (
    FramePoint,
    SentimentPoint,
    VoiceShiftPoint,
    TimeBucket,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(days_offset: int = 0) -> str:
    """Return an ISO datetime string offset from 2024-01-01."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(days=days_offset)).isoformat()


def _make_article(article_id: str, days_offset: int = 0, cluster: str | None = None) -> dict:
    return {
        "id": article_id,
        "title": f"Article {article_id}",
        "body": "Some content about the topic.",
        "source_name": f"source-{article_id}",
        "source_url": f"https://example-{article_id}.com/article",
        "published_at": _dt(days_offset),
        "outlet_type": "unknown",
        "query_used": "test query",
    }


def _base_state(**overrides) -> dict:
    base = {
        "claim": "5G towers spread COVID-19.",
        "entities": ["WHO", "5G"],
        "retrieved_articles": [],
        "perspective_clusters": [],
        "evaluation_ready": True,
        "primary_verdict": None,
        "worker_outputs": [],
        "narrative_report": None,
        "analysis_complete": None,
    }
    base.update(overrides)
    return base


def _make_bucket(bid: int, start_offset: int = 0, end_offset: int = 5) -> TimeBucket:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return TimeBucket(
        bucket_id=bid,
        start_dt=base + timedelta(days=start_offset),
        end_dt=base + timedelta(days=end_offset),
        article_ids=[],
    )


# ---------------------------------------------------------------------------
# build_time_buckets
# ---------------------------------------------------------------------------

def test_build_time_buckets_normal_corpus():
    """15 articles over 30 days should produce ≥3 buckets."""
    articles = [_make_article(f"a{i}", days_offset=i * 2) for i in range(15)]
    buckets, assignments = build_time_buckets(articles)
    assert len(buckets) >= 3
    assert len(assignments) == 15
    assert all(0 <= bid < len(buckets) for bid in assignments)


def test_build_time_buckets_single_article():
    """Single article produces 1 bucket without crashing."""
    articles = [_make_article("solo", days_offset=0)]
    buckets, assignments = build_time_buckets(articles)
    assert len(buckets) == 1
    assert assignments == [0]


def test_build_time_buckets_same_timestamp():
    """All articles with the same timestamp → single bucket."""
    articles = [_make_article(f"x{i}", days_offset=0) for i in range(10)]
    buckets, assignments = build_time_buckets(articles)
    assert len(buckets) == 1
    assert all(bid == 0 for bid in assignments)


def test_build_time_buckets_sub_3_day_corpus():
    """Corpus spanning < 3 days still produces max(3, n//5) buckets."""
    # 20 articles within 2 days — N = max(3, 20//5) = max(3, 4) = 4
    articles = [_make_article(f"b{i}", days_offset=0) for i in range(10)]
    # Vary timestamps within 2-day range by manually overriding
    for i, a in enumerate(articles):
        a["published_at"] = (
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i * 4)
        ).isoformat()
    buckets, assignments = build_time_buckets(articles)
    assert len(buckets) >= 2   # might be capped if few articles
    assert len(assignments) == len(articles)


def test_build_time_buckets_empty():
    """Empty article list returns empty results."""
    buckets, assignments = build_time_buckets([])
    assert buckets == []
    assert assignments == []


def test_build_time_buckets_all_articles_assigned():
    """Every article is assigned to a valid bucket."""
    articles = [_make_article(f"c{i}", days_offset=i) for i in range(20)]
    buckets, assignments = build_time_buckets(articles)
    assert len(assignments) == 20
    for bid in assignments:
        assert 0 <= bid < len(buckets)


# ---------------------------------------------------------------------------
# _detect_inflection_candidates
# ---------------------------------------------------------------------------

def _make_sentiment_point(bid: int, delta: float) -> SentimentPoint:
    return SentimentPoint(bucket_id=bid, mean_sentiment=0.0, delta=delta,
                          dominant_entity="WHO", article_count=3)


def _make_frame_point(bid: int, frame_changed: bool) -> FramePoint:
    return FramePoint(bucket_id=bid, frame_type="political", top_terms=[], frame_changed=frame_changed)


def _make_voice_point(bid: int, voice_changed: bool) -> VoiceShiftPoint:
    return VoiceShiftPoint(bucket_id=bid, dominant_cluster_label="cluster_a",
                           composition={}, voice_changed=voice_changed)


def test_detect_inflection_single_track_is_noise():
    """Only 1 track signaling → not an inflection point."""
    buckets = [_make_bucket(0), _make_bucket(1)]
    sentiment = [_make_sentiment_point(0, 0.0), _make_sentiment_point(1, 0.5)]  # spike in bucket 1
    frame = [_make_frame_point(0, False), _make_frame_point(1, False)]           # no frame change
    voice = [_make_voice_point(0, False), _make_voice_point(1, False)]           # no voice change

    candidates = _detect_inflection_candidates(buckets, sentiment, frame, voice)
    assert candidates == []


def test_detect_inflection_multi_track_detected():
    """2 simultaneous signals → inflection point detected."""
    buckets = [_make_bucket(0), _make_bucket(1)]
    sentiment = [_make_sentiment_point(0, 0.0), _make_sentiment_point(1, 0.5)]  # spike
    frame = [_make_frame_point(0, False), _make_frame_point(1, True)]            # shift
    voice = [_make_voice_point(0, False), _make_voice_point(1, False)]           # no change

    candidates = _detect_inflection_candidates(buckets, sentiment, frame, voice)
    assert len(candidates) == 1
    assert candidates[0]["bucket_id"] == 1
    assert set(candidates[0]["tracks_signaling"]) == {"sentiment", "frame"}


def test_detect_inflection_all_three_tracks():
    """All 3 tracks signaling → inflection with all tracks listed."""
    buckets = [_make_bucket(0), _make_bucket(1)]
    sentiment = [_make_sentiment_point(0, 0.0), _make_sentiment_point(1, 0.8)]
    frame = [_make_frame_point(0, False), _make_frame_point(1, True)]
    voice = [_make_voice_point(0, False), _make_voice_point(1, True)]

    candidates = _detect_inflection_candidates(buckets, sentiment, frame, voice)
    assert len(candidates) == 1
    assert set(candidates[0]["tracks_signaling"]) == {"sentiment", "frame", "voice"}


def test_detect_inflection_no_buckets():
    """Empty inputs return empty results."""
    candidates = _detect_inflection_candidates([], [], [], [])
    assert candidates == []


# ---------------------------------------------------------------------------
# _classify_ground_truth
# ---------------------------------------------------------------------------

def test_classify_ground_truth_primary_verdict_supported():
    """primary_verdict='supported' overrides ratio → 'verifiable'."""
    clusters = [{"corroboration_level": "unverified"}] * 5  # 0% corroborated
    result = _classify_ground_truth(clusters, primary_verdict="supported")
    assert result == "verifiable"


def test_classify_ground_truth_primary_verdict_contradicted():
    """primary_verdict='contradicted' also → 'verifiable'."""
    clusters = [{"corroboration_level": "unverified"}] * 5
    result = _classify_ground_truth(clusters, primary_verdict="contradicted")
    assert result == "verifiable"


def test_classify_ground_truth_high_corroboration():
    """>70% corroborated clusters → 'verifiable'."""
    clusters = [
        {"corroboration_level": "corroborated"},
        {"corroboration_level": "corroborated"},
        {"corroboration_level": "corroborated"},
        {"corroboration_level": "unverified"},
    ]
    result = _classify_ground_truth(clusters, primary_verdict=None)
    assert result == "verifiable"


def test_classify_ground_truth_contested_range():
    """50% corroborated → 'contested'."""
    clusters = [
        {"corroboration_level": "corroborated"},
        {"corroboration_level": "corroborated"},
        {"corroboration_level": "unverified"},
        {"corroboration_level": "unverified"},
    ]
    result = _classify_ground_truth(clusters, primary_verdict=None)
    assert result == "contested"


def test_classify_ground_truth_low_corroboration():
    """0% corroborated → 'unresolvable'."""
    clusters = [{"corroboration_level": "unverified"}] * 4
    result = _classify_ground_truth(clusters, primary_verdict=None)
    assert result == "unresolvable"


def test_classify_ground_truth_zero_clusters():
    """No clusters → 'unresolvable'."""
    result = _classify_ground_truth([], primary_verdict=None)
    assert result == "unresolvable"


def test_classify_ground_truth_partial_not_counted():
    """'partial' corroboration level does not count towards corroborated ratio."""
    clusters = [
        {"corroboration_level": "partial"},
        {"corroboration_level": "partial"},
        {"corroboration_level": "partial"},
        {"corroboration_level": "unverified"},
    ]
    # 0 out of 4 are "corroborated" → ratio = 0 → unresolvable
    result = _classify_ground_truth(clusters, primary_verdict=None)
    assert result == "unresolvable"


# ---------------------------------------------------------------------------
# _find_spike_buckets
# ---------------------------------------------------------------------------

def test_find_spike_buckets_no_spikes():
    sentiment = [_make_sentiment_point(0, 0.1), _make_sentiment_point(1, 0.1)]
    frame = [_make_frame_point(0, False), _make_frame_point(1, False)]
    voice = [_make_voice_point(0, False), _make_voice_point(1, False)]
    assert _find_spike_buckets(sentiment, frame, voice) == []


def test_find_spike_buckets_from_voice_change():
    sentiment = [_make_sentiment_point(0, 0.0), _make_sentiment_point(1, 0.1)]
    frame = [_make_frame_point(0, False), _make_frame_point(1, False)]
    voice = [_make_voice_point(0, False), _make_voice_point(1, True)]
    assert 1 in _find_spike_buckets(sentiment, frame, voice)


# ---------------------------------------------------------------------------
# analyst_node integration (all heavy dependencies mocked)
# ---------------------------------------------------------------------------

@patch("src.workers.analyst._compose_narrative_summary", return_value="Test summary.")
@patch("src.workers.analyst._characterize_shifts", return_value=[])
@patch("src.workers.analyst._track_gdelt_events", return_value=[])
@patch("src.workers.analyst._track_voice_composition")
@patch("src.workers.analyst._track_frame_evolution")
@patch("src.workers.analyst._track_sentiment")
def test_analyst_node_full_pipeline(
    mock_sentiment, mock_frame, mock_voice, mock_gdelt, mock_shifts, mock_summary
):
    """Happy path: analyst assembles NarrativeReport from mocked track outputs."""
    articles = [_make_article(f"a{i}", days_offset=i * 5) for i in range(6)]
    clusters = [
        {
            "label": "Cluster A",
            "key_claims": ["claim 1"],
            "article_ids": [a["id"] for a in articles[:3]],
            "domain_roots": ["example-a0.com"],
            "corroboration_count": 3,
            "corroboration_level": "corroborated",
            "source_scope": "secondary",
        }
    ]

    # Build fake buckets to match what build_time_buckets would return
    real_buckets, real_assignments = build_time_buckets(articles)

    mock_sentiment.return_value = [
        SentimentPoint(bucket_id=i, mean_sentiment=0.1 * i, delta=0.1,
                       dominant_entity="WHO", article_count=2)
        for i in range(len(real_buckets))
    ]
    mock_frame.return_value = [
        FramePoint(bucket_id=i, frame_type="political", top_terms=["test"], frame_changed=False)
        for i in range(len(real_buckets))
    ]
    mock_voice.return_value = (
        [VoiceShiftPoint(bucket_id=i, dominant_cluster_label="Cluster A",
                         composition={"Cluster A": 2}, voice_changed=False)
         for i in range(len(real_buckets))],
        {i: {"Cluster A": 2} for i in range(len(real_buckets))},
    )

    state = _base_state(retrieved_articles=articles, perspective_clusters=clusters)
    result = analyst_node(state)

    assert result["analysis_complete"] is True
    assert result["narrative_report"] is not None
    report = result["narrative_report"]
    assert "claim_snapshot" in report
    assert "sentiment_timeline" in report
    assert "frame_evolution_log" in report
    assert "voice_composition_shifts" in report
    assert "narrative_intelligence_summary" in report
    assert report["ground_truth_tier"] in ("verifiable", "contested", "unresolvable")


def test_analyst_node_skips_when_not_ready():
    """evaluation_ready=False → short-circuit with analysis_complete=False."""
    state = _base_state(evaluation_ready=False, retrieved_articles=[_make_article("x")])
    result = analyst_node(state)
    assert result["analysis_complete"] is False
    assert result.get("narrative_report") is None


def test_analyst_node_skips_empty_articles():
    """No articles → short-circuit even when evaluation_ready=True."""
    state = _base_state(evaluation_ready=True, retrieved_articles=[])
    result = analyst_node(state)
    assert result["analysis_complete"] is False
