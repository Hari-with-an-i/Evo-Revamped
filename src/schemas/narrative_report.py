from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TimeBucket(BaseModel):
    bucket_id: int
    start_dt: datetime
    end_dt: datetime
    article_ids: list[str]


class SentimentPoint(BaseModel):
    bucket_id: int
    mean_sentiment: float = Field(ge=-1.0, le=1.0)
    delta: float          # 0.0 for bucket_id == 0
    dominant_entity: str
    article_count: int


class FramePoint(BaseModel):
    bucket_id: int
    frame_type: Literal["scientific", "political", "economic", "legal", "cultural", "health", "other"]
    top_terms: list[str]  # top 20 TF-IDF terms for this bucket
    frame_changed: bool   # True when frame_type differs from previous bucket


class VoiceShiftPoint(BaseModel):
    bucket_id: int
    dominant_cluster_label: str
    composition: dict[str, int]  # {cluster_label: article_count}
    voice_changed: bool           # True when dominant cluster differs from previous bucket


class GDELTEvent(BaseModel):
    event_date: str
    description: str
    plausibility_score: float = Field(ge=0.0, le=1.0)
    correlated_bucket: int


class InflectionPoint(BaseModel):
    bucket_id: int
    bucket_date_range: str         # e.g. "2024-01-05 to 2024-01-10"
    tracks_signaling: list[str]    # e.g. ["sentiment", "frame", "voice"]
    correlated_event: GDELTEvent | None
    dominant_cluster_label: str
    explanation: str               # LLM-generated shift characterization (≤3 sentences)


class NarrativeReport(BaseModel):
    # Section 1
    claim_snapshot: dict           # {claim, time_period, source_count, ground_truth_tier}
    # Section 2
    perspective_landscape: list[dict]
    # Section 3
    sentiment_timeline: list[SentimentPoint]
    # Section 4
    inflection_point_cards: list[InflectionPoint]
    # Section 5
    frame_evolution_log: list[FramePoint]
    # Section 6
    voice_composition_shifts: list[VoiceShiftPoint]
    # Section 7
    narrative_intelligence_summary: str
    # Metadata
    ground_truth_tier: Literal["verifiable", "contested", "unresolvable"]
    time_buckets: list[TimeBucket]
