"""
Analyst worker — narrative intelligence analysis pipeline.

Runs as a direct edge after evaluator_node, before returning to orchestrator.
Implements four parallel analysis tracks, inflection detection, shift characterization,
ground truth classification, and final report assembly.

Entry point: analyst_node(state: AgentState) -> dict
"""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import timedelta
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import config
from src.llm import get_classification_llm
from src.logger import get_logger, timer
from src.schemas.narrative_report import (
    FramePoint,
    GDELTEvent,
    InflectionPoint,
    NarrativeReport,
    SentimentPoint,
    TimeBucket,
    VoiceShiftPoint,
)
from src.state import AgentState
from src.utils.bucketing import build_time_buckets

log = get_logger(__name__)

SENTIMENT_DELTA_THRESHOLD: float = config.SENTIMENT_DELTA_THRESHOLD
MIN_INFLECTION_TRACKS = 2          # minimum simultaneous track signals for an inflection point


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    return ChatGroq(model=config.ANALYST_MODEL_NAME, api_key=config.GROQ_API_KEY, max_tokens=600)


def _get_small_llm():
    return get_classification_llm()


class _FrameClassification(BaseModel):
    frame_type: Literal[
        "scientific", "political", "economic", "legal", "cultural", "health", "other"
    ] = Field(description="The dominant narrative frame suggested by these terms.")


class _BatchFrameClassification(BaseModel):
    frame_types: list[str] = Field(
        description=(
            "One frame label per bucket in order. Each label must be one of: "
            "scientific, political, economic, legal, cultural, health, other."
        )
    )


class _CausalPlausibility(BaseModel):
    plausibility_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = clearly unrelated, 1.0 = strongly plausible causal link.",
    )
    reasoning: str = Field(description="One sentence explaining the score.")


class _BestEventPick(BaseModel):
    best_index: int = Field(
        description="0-based index of the most causally plausible event from the list, or -1 if none are plausible."
    )
    plausibility_score: float = Field(
        ge=0.0, le=1.0,
        description="Plausibility score for the chosen event (0.0–1.0).",
    )


class _ShiftExplanation(BaseModel):
    explanation: str = Field(
        description="2–3 sentences: what changed, which perspective gained dominance, why it matters."
    )


class _NarrativeSummary(BaseModel):
    summary: str = Field(
        description="Up to 3 paragraphs of narrative intelligence summary appropriate to the ground truth tier."
    )


# ---------------------------------------------------------------------------
# Track 1 — Sentiment trajectory
# ---------------------------------------------------------------------------

def _extract_entity_sentences(article: dict, entities: list[str]) -> tuple[list[str], str]:
    """
    Split article body into sentences, return those mentioning any entity.
    Fallback: first 3 sentences if none match. Also returns the dominant entity.
    """
    body: str = article.get("body", "") or ""
    raw_sentences = [s.strip() for s in body.split(". ") if s.strip()]
    sentences = raw_sentences or []

    if not entities:
        return sentences[:3], "unknown"

    entity_lower = [e.lower() for e in entities]
    matching: list[str] = []
    entity_counts: Counter = Counter()

    for sent in sentences:
        sent_lower = sent.lower()
        for ent, ent_low in zip(entities, entity_lower):
            if ent_low in sent_lower:
                matching.append(sent)
                entity_counts[ent] += 1
                break  # count sentence once even if multiple entities match

    if not matching:
        matching = sentences[:3]
        # dominant entity: fallback to full-body scan
        for ent, ent_low in zip(entities, entity_lower):
            entity_counts[ent] = body.lower().count(ent_low)

    dominant = entity_counts.most_common(1)[0][0] if entity_counts else (entities[0] if entities else "unknown")
    return matching, dominant


def _track_sentiment(
    articles: list[dict],
    assignments: list[int],
    buckets: list[TimeBucket],
    entities: list[str],
) -> list[SentimentPoint]:
    from src.utils.roberta_sentiment import score_sentiment  # lazy import

    bucket_sentences: list[list[str]] = [[] for _ in buckets]
    bucket_entity_counts: list[Counter] = [Counter() for _ in buckets]

    for i, article in enumerate(articles):
        bid = assignments[i]
        sents, dominant = _extract_entity_sentences(article, entities)
        bucket_sentences[bid].extend(sents)
        bucket_entity_counts[bid][dominant] += len(sents)

    prev_mean = 0.0
    points: list[SentimentPoint] = []

    for b, bucket in enumerate(buckets):
        sents = bucket_sentences[b]
        article_count = sum(1 for a in assignments if a == b)

        if not sents:
            mean_sent = 0.0
        else:
            try:
                scores = score_sentiment(sents)
                mean_sent = statistics.mean(scores) if scores else 0.0
            except Exception:
                log.warning("sentiment scoring failed for bucket %d", b)
                mean_sent = 0.0

        delta = mean_sent - prev_mean if b > 0 else 0.0
        counts = bucket_entity_counts[b]
        dominant_entity = counts.most_common(1)[0][0] if counts else (entities[0] if entities else "unknown")

        points.append(SentimentPoint(
            bucket_id=b,
            mean_sentiment=round(max(-1.0, min(1.0, mean_sent)), 4),
            delta=round(delta, 4),
            dominant_entity=dominant_entity,
            article_count=article_count,
        ))
        prev_mean = mean_sent

    return points


# ---------------------------------------------------------------------------
# Track 2 — Frame evolution
# ---------------------------------------------------------------------------

_VALID_FRAMES = {"scientific", "political", "economic", "legal", "cultural", "health", "other"}


def _track_frame_evolution(
    articles: list[dict],
    assignments: list[int],
    buckets: list[TimeBucket],
) -> list[FramePoint]:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

    # Pass 1: compute TF-IDF top terms per bucket (no LLM)
    bucket_top_terms: list[list[str]] = []
    for b in range(len(buckets)):
        bucket_bodies = [
            (articles[i].get("body") or "")[:500]
            for i, bid in enumerate(assignments)
            if bid == b
        ]
        top_terms: list[str] = []
        if bucket_bodies:
            try:
                vec = TfidfVectorizer(stop_words="english", max_features=500)
                tfidf = vec.fit_transform(bucket_bodies)
                import numpy as np  # type: ignore[import]
                mean_scores = tfidf.mean(axis=0).A1
                top_indices = mean_scores.argsort()[::-1][:20]
                feature_names = vec.get_feature_names_out()
                top_terms = [feature_names[i] for i in top_indices]
            except Exception:
                log.warning("TF-IDF failed for bucket %d", b)
        bucket_top_terms.append(top_terms)

    # Pass 2: single batched LLM call for all buckets that have terms
    buckets_with_terms = [(b, terms) for b, terms in enumerate(bucket_top_terms) if terms]
    frame_map: dict[int, str] = {}
    if buckets_with_terms:
        bucket_lines = "\n".join(
            f"Bucket {b}: {', '.join(terms[:20])}"
            for b, terms in buckets_with_terms
        )
        llm = _get_small_llm().with_structured_output(_BatchFrameClassification)
        try:
            with timer(log, "frame_llm_batch", buckets=len(buckets_with_terms)):
                batch_result: _BatchFrameClassification = llm.invoke([
                    SystemMessage(content=(
                        "You are a media framing analyst. "
                        "For each bucket's top terms, classify the dominant narrative frame. "
                        "Valid frames: scientific, political, economic, legal, cultural, health, other. "
                        "Return exactly one label per bucket in order."
                    )),
                    HumanMessage(content=f"Buckets:\n{bucket_lines}"),
                ])
            for (b, _), label in zip(buckets_with_terms, batch_result.frame_types):
                frame_map[b] = label if label in _VALID_FRAMES else "other"
        except Exception:
            log.warning("batch frame classification LLM failed — defaulting to 'other'")

    # Assemble FramePoint list
    prev_frame: str | None = None
    points: list[FramePoint] = []
    for b, bucket in enumerate(buckets):
        frame_type = frame_map.get(b, "other")
        frame_changed = (prev_frame is not None) and (frame_type != prev_frame)
        points.append(FramePoint(
            bucket_id=b,
            frame_type=frame_type,
            top_terms=bucket_top_terms[b],
            frame_changed=frame_changed,
        ))
        prev_frame = frame_type

    return points


# ---------------------------------------------------------------------------
# Track 3 — Voice composition
# ---------------------------------------------------------------------------

def _assign_articles_to_clusters(
    articles: list[dict],
    perspective_clusters: list[dict],
) -> dict[str, str]:
    """Map article_id → cluster_label. Unmapped articles → '_unclustered'."""
    mapping: dict[str, str] = {}
    for cluster in perspective_clusters:
        label = cluster.get("label", "_unclustered")
        for art_id in cluster.get("article_ids", []):
            mapping[art_id] = label
    for article in articles:
        art_id = article.get("id", "")
        if art_id not in mapping:
            mapping[art_id] = "_unclustered"
    return mapping


def _track_voice_composition(
    articles: list[dict],
    assignments: list[int],
    buckets: list[TimeBucket],
    perspective_clusters: list[dict],
) -> tuple[list[VoiceShiftPoint], dict[int, dict[str, int]]]:
    art_to_cluster = _assign_articles_to_clusters(articles, perspective_clusters)

    prev_dominant: str | None = None
    points: list[VoiceShiftPoint] = []
    composition_map: dict[int, dict[str, int]] = {}

    for b, bucket in enumerate(buckets):
        counts: Counter = Counter()
        for i, bid in enumerate(assignments):
            if bid == b:
                art_id = articles[i].get("id", "")
                cluster_label = art_to_cluster.get(art_id, "_unclustered")
                counts[cluster_label] += 1

        composition = dict(counts)
        composition_map[b] = composition

        if counts:
            # stable sort: most common first, tie-broken by cluster index order
            dominant = counts.most_common(1)[0][0]
        else:
            dominant = "_unclustered"

        voice_changed = (prev_dominant is not None) and (dominant != prev_dominant)
        points.append(VoiceShiftPoint(
            bucket_id=b,
            dominant_cluster_label=dominant,
            composition=composition,
            voice_changed=voice_changed,
        ))
        prev_dominant = dominant

    return points, composition_map


# ---------------------------------------------------------------------------
# Track 4 — GDELT events
# ---------------------------------------------------------------------------

def _find_spike_buckets(
    sentiment_points: list[SentimentPoint],
    frame_points: list[FramePoint],
    voice_points: list[VoiceShiftPoint],
    delta_threshold: float = SENTIMENT_DELTA_THRESHOLD,
) -> list[int]:
    """Return bucket_ids that show any single-track signal (used to trigger GDELT queries)."""
    spikes: set[int] = set()
    for sp in sentiment_points:
        if abs(sp.delta) > delta_threshold:
            spikes.add(sp.bucket_id)
    for fp in frame_points:
        if fp.frame_changed:
            spikes.add(fp.bucket_id)
    for vp in voice_points:
        if vp.voice_changed:
            spikes.add(vp.bucket_id)
    return sorted(spikes)


def _track_gdelt_events(
    buckets: list[TimeBucket],
    spike_bucket_ids: list[int],
    entities: list[str],
    claim: str,
) -> list[GDELTEvent]:
    if not spike_bucket_ids:
        return []

    from src.tools.news_tools import gdelt_search  # avoid circular at module level

    # One LLM call per bucket: ask model to pick the best candidate index rather than
    # scoring each candidate individually (reduces N calls to 1 per spike bucket).
    llm = _get_small_llm().with_structured_output(_BestEventPick)
    events: list[GDELTEvent] = []

    claim_words = claim.split()[:5]
    entity_terms = " ".join(entities[:2])
    query = f"{entity_terms} {' '.join(claim_words)}".strip()

    for bid in spike_bucket_ids:
        bucket = buckets[bid]
        window_start = bucket.start_dt - timedelta(days=3)
        window_end = bucket.end_dt + timedelta(days=3)
        span_days = max(1, (window_end - window_start).days)
        timespan = f"{span_days}d"

        try:
            raw_results: list[dict] = gdelt_search.invoke({
                "query": query,
                "timespan": timespan,
                "max_records": 5,
            })
        except Exception:
            log.warning("GDELT query failed for bucket %d spike", bid)
            continue

        candidates = [
            r for r in raw_results
            if not r.get("_error") and r.get("title")
        ]
        if not candidates:
            continue

        event_lines = "\n".join(
            f"{i}. [{r.get('seendate', '')}] {r.get('title', '')}"
            for i, r in enumerate(candidates)
        )
        try:
            with timer(log, "gdelt_plausibility_batch", bucket=bid, candidates=len(candidates)):
                pick: _BestEventPick = llm.invoke([
                    SystemMessage(content=(
                        "You are a media analyst assessing causal plausibility. "
                        "Pick the single most plausible event that could have caused or coincided with "
                        "a measurable shift in public narrative about the given claim. "
                        "Return -1 if none are plausible."
                    )),
                    HumanMessage(content=(
                        f"Claim: {claim}\n\n"
                        f"Candidate events:\n{event_lines}"
                    )),
                ])
            idx = pick.best_index
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                events.append(GDELTEvent(
                    event_date=chosen.get("seendate", ""),
                    description=chosen.get("title", ""),
                    plausibility_score=round(pick.plausibility_score, 3),
                    correlated_bucket=bid,
                ))
        except Exception:
            log.warning("GDELT plausibility LLM failed for bucket %d", bid)

    return events


# ---------------------------------------------------------------------------
# Inflection point detection
# ---------------------------------------------------------------------------

def _detect_inflection_candidates(
    buckets: list[TimeBucket],
    sentiment_points: list[SentimentPoint],
    frame_points: list[FramePoint],
    voice_points: list[VoiceShiftPoint],
    delta_threshold: float = SENTIMENT_DELTA_THRESHOLD,
    min_tracks: int = MIN_INFLECTION_TRACKS,
) -> list[dict]:
    """
    Return buckets where ≥ min_tracks signals fire simultaneously.
    Single-track spikes are noise and excluded.
    """
    sp_map = {p.bucket_id: p for p in sentiment_points}
    fp_map = {p.bucket_id: p for p in frame_points}
    vp_map = {p.bucket_id: p for p in voice_points}

    candidates: list[dict] = []
    for bucket in buckets:
        bid = bucket.bucket_id
        tracks: list[str] = []

        sp = sp_map.get(bid)
        if sp and abs(sp.delta) > delta_threshold:
            tracks.append("sentiment")

        fp = fp_map.get(bid)
        if fp and fp.frame_changed:
            tracks.append("frame")

        vp = vp_map.get(bid)
        if vp and vp.voice_changed:
            tracks.append("voice")

        if len(tracks) >= min_tracks:
            date_range = f"{bucket.start_dt.date()} to {bucket.end_dt.date()}"
            candidates.append({
                "bucket_id": bid,
                "tracks_signaling": tracks,
                "bucket_date_range": date_range,
            })

    return candidates


# ---------------------------------------------------------------------------
# Shift characterizer
# ---------------------------------------------------------------------------

def _characterize_shifts(
    candidates: list[dict],
    buckets: list[TimeBucket],
    sentiment_points: list[SentimentPoint],
    frame_points: list[FramePoint],
    voice_points: list[VoiceShiftPoint],
    gdelt_events: list[GDELTEvent],
    perspective_clusters: list[dict],
    claim: str,
) -> list[InflectionPoint]:
    if not candidates:
        return []

    llm = _get_small_llm().with_structured_output(_ShiftExplanation)

    gdelt_map: dict[int, GDELTEvent] = {e.correlated_bucket: e for e in gdelt_events}
    sp_map = {p.bucket_id: p for p in sentiment_points}
    fp_map = {p.bucket_id: p for p in frame_points}
    vp_map = {p.bucket_id: p for p in voice_points}

    inflection_points: list[InflectionPoint] = []

    for cand in candidates:
        bid = cand["bucket_id"]
        sp = sp_map.get(bid)
        fp = fp_map.get(bid)
        vp = vp_map.get(bid)
        event = gdelt_map.get(bid)

        context_parts = [f"Claim: {claim}", f"Bucket: {cand['bucket_date_range']}"]
        if sp:
            context_parts.append(
                f"Sentiment shift: {sp.delta:+.2f} (entity: {sp.dominant_entity})"
            )
        if fp:
            context_parts.append(
                f"Frame changed to '{fp.frame_type}' (top terms: {', '.join(fp.top_terms[:8])})"
            )
        if vp:
            context_parts.append(
                f"Dominant voice shifted to '{vp.dominant_cluster_label}'"
            )
        if event:
            context_parts.append(
                f"Correlated event ({event.event_date}): {event.description}"
            )

        explanation = "(characterization unavailable)"
        try:
            with timer(log, "shift_characterizer", bucket=bid):
                result: _ShiftExplanation = llm.invoke([
                    SystemMessage(content=(
                        "You are a narrative analyst. Given signals from multiple analysis tracks, "
                        "explain in 2–3 sentences what changed in the media narrative, which perspective "
                        "gained dominance, and why this shift matters."
                    )),
                    HumanMessage(content="\n".join(context_parts)),
                ])
            explanation = result.explanation
        except Exception:
            log.warning("shift characterizer LLM failed for bucket %d", bid)

        dominant_cluster = vp.dominant_cluster_label if vp else "_unclustered"

        inflection_points.append(InflectionPoint(
            bucket_id=bid,
            bucket_date_range=cand["bucket_date_range"],
            tracks_signaling=cand["tracks_signaling"],
            correlated_event=event,
            dominant_cluster_label=dominant_cluster,
            explanation=explanation,
        ))

    return inflection_points


# ---------------------------------------------------------------------------
# Ground truth classifier
# ---------------------------------------------------------------------------

def _classify_ground_truth(
    perspective_clusters: list[dict],
    primary_verdict: str | None,
) -> Literal["verifiable", "contested", "unresolvable"]:
    # Primary evidence overrides ratio calculation
    if primary_verdict in ("supported", "contradicted"):
        return "verifiable"

    if not perspective_clusters:
        return "unresolvable"

    corroborated_count = sum(
        1 for c in perspective_clusters
        if c.get("corroboration_level") == "corroborated"
    )
    ratio = corroborated_count / len(perspective_clusters)

    if ratio > 0.7:
        return "verifiable"
    elif ratio >= 0.4:
        return "contested"
    else:
        return "unresolvable"


# ---------------------------------------------------------------------------
# Narrative summary composer
# ---------------------------------------------------------------------------

def _compose_narrative_summary(
    ground_truth_tier: str,
    perspective_clusters: list[dict],
    inflection_points: list[InflectionPoint],
    claim: str,
    primary_verdict: str | None,
) -> str:
    llm = _get_llm().with_structured_output(_NarrativeSummary)

    cluster_summaries = "\n".join(
        f"- {c.get('label', '?')} "
        f"[{c.get('corroboration_level', '?')}, "
        f"{c.get('corroboration_count', 0)} domains]"
        for c in perspective_clusters[:10]
    )

    inflection_summaries = "\n".join(
        f"- Bucket {ip.bucket_id} ({ip.bucket_date_range}): {ip.explanation}"
        for ip in inflection_points[:5]
    ) or "No multi-track inflection points detected."

    if ground_truth_tier == "verifiable":
        task = (
            "Produce a verifiable ground truth statement. "
            "Cite the corroboration count and specific domain roots. "
            "Note whether primary evidence supports or contradicts the claim. "
            "Explicitly support or debunk the statement based on the facts. "
            "If the evidence points to a future trend, provide a forecast based on these facts."
        )
    elif ground_truth_tier == "contested":
        task = (
            "Summarise the contested landscape. "
            "For each major perspective, state what it claims and where it diverges "
            "from the others. Based on the facts, evaluate which side has stronger empirical backing "
            "to support or debunk the core statement, if possible. "
            "If the evidence points to a future trend, provide a forecast based on these facts."
        )
    else:
        task = (
            "Flag explicitly: the evidence is insufficient or conflicting to reach a verdict. "
            "Present all known perspectives without resolution. "
            "Explain why a definitive verdict is not possible. "
            "Provide a forecast on how this narrative might evolve based on current facts."
        )

    prompt = (
        f"Claim: {claim}\n"
        f"Ground truth tier: {ground_truth_tier}\n"
        f"Primary verdict: {primary_verdict or 'none'}\n\n"
        f"Perspective clusters:\n{cluster_summaries}\n\n"
        f"Key narrative shifts:\n{inflection_summaries}\n\n"
        f"Task: {task}"
    )

    try:
        with timer(log, "narrative_summary"):
            result: _NarrativeSummary = llm.invoke([
                SystemMessage(content=(
                    "You are a senior investigative journalist writing a narrative intelligence brief. "
                    "Write no more than 3 paragraphs. Be factual, precise, and explicitly analytical."
                )),
                HumanMessage(content=prompt),
            ])
        return result.summary
    except Exception:
        log.warning("narrative summary LLM failed")
        return "(narrative summary unavailable)"


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

def analyst_node(state: AgentState) -> dict:
    """
    Narrative analysis pipeline node.

    Runs after evaluator via direct graph edge. Short-circuits if evaluation_ready
    is False or there are no articles (passes analysis_complete=False back to
    orchestrator, which continues retrieval looping as normal).
    """
    evaluation_ready: bool | None = state.get("evaluation_ready")
    articles: list[dict] = state.get("retrieved_articles") or []
    clusters: list[dict] = state.get("perspective_clusters") or []
    claim: str = state.get("claim") or ""
    entities: list[str] = state.get("entities") or []
    primary_verdict: str | None = state.get("primary_verdict")

    if evaluation_ready is None or not articles:
        log.info("analyst skipped — evaluation_ready=%s, articles=%d", evaluation_ready, len(articles))
        return {
            "analysis_complete": False,
            "worker_outputs": ["[analyst]\nSkipped — evaluator has not run yet or no articles."],
        }

    log.info("analyst starting — %d articles, %d clusters", len(articles), len(clusters))

    # Step 1: time bucketing
    context_summary = state.get("context_summary", {})
    predefined_periods = context_summary.get("relevant_time_periods") if isinstance(context_summary, dict) else None
    buckets, assignments = build_time_buckets(articles, predefined_periods=predefined_periods)
    log.info("analyst: %d time buckets created", len(buckets))

    # Step 2: Track 1 — sentiment
    try:
        sentiment_curve = _track_sentiment(articles, assignments, buckets, entities)
    except Exception:
        log.exception("Track 1 (sentiment) failed — using zero baseline")
        sentiment_curve = [
            SentimentPoint(bucket_id=b.bucket_id, mean_sentiment=0.0, delta=0.0,
                           dominant_entity="unknown", article_count=0)
            for b in buckets
        ]

    # Step 3: Track 2 — frame evolution
    try:
        frame_sequence = _track_frame_evolution(articles, assignments, buckets)
    except Exception:
        log.exception("Track 2 (frame) failed — using 'other' baseline")
        frame_sequence = [
            FramePoint(bucket_id=b.bucket_id, frame_type="other", top_terms=[], frame_changed=False)
            for b in buckets
        ]

    # Step 4: Track 3 — voice composition
    try:
        voice_shifts, composition_map = _track_voice_composition(
            articles, assignments, buckets, clusters
        )
    except Exception:
        log.exception("Track 3 (voice) failed — using empty baseline")
        voice_shifts = [
            VoiceShiftPoint(bucket_id=b.bucket_id, dominant_cluster_label="_unclustered",
                            composition={}, voice_changed=False)
            for b in buckets
        ]
        composition_map = {}

    # Step 5: Track 4 — GDELT events near spikes
    spike_ids = _find_spike_buckets(sentiment_curve, frame_sequence, voice_shifts)
    try:
        gdelt_events = _track_gdelt_events(buckets, spike_ids, entities, claim)
    except Exception:
        log.exception("Track 4 (GDELT) failed — continuing without events")
        gdelt_events = []

    # Step 6: Inflection point detection
    candidates = _detect_inflection_candidates(buckets, sentiment_curve, frame_sequence, voice_shifts)
    log.info("analyst: %d inflection candidates detected", len(candidates))

    # Step 7: Shift characterization (LLM per inflection)
    try:
        inflection_points = _characterize_shifts(
            candidates, buckets, sentiment_curve, frame_sequence,
            voice_shifts, gdelt_events, clusters, claim,
        )
    except Exception:
        log.exception("Shift characterization failed")
        inflection_points = []

    # Step 8: Ground truth classification
    ground_truth_tier = _classify_ground_truth(clusters, primary_verdict)
    log.info("analyst: ground_truth_tier=%s", ground_truth_tier)

    # Step 9: Narrative summary (single LLM call)
    narrative_summary = _compose_narrative_summary(
        ground_truth_tier, clusters, inflection_points, claim, primary_verdict
    )

    # Build perspective landscape (clusters enriched with article_count per cluster)
    art_to_cluster = _assign_articles_to_clusters(articles, clusters)
    cluster_article_counts: Counter = Counter(art_to_cluster.values())
    perspective_landscape = []
    for c in clusters:
        enriched = dict(c)
        enriched["article_count"] = cluster_article_counts.get(c.get("label", ""), 0)
        perspective_landscape.append(enriched)

    # Assemble report
    time_period = (
        f"{buckets[0].start_dt.date()} to {buckets[-1].end_dt.date()}"
        if buckets else "unknown"
    )
    report = NarrativeReport(
        claim_snapshot={
            "claim": claim,
            "time_period": time_period,
            "source_count": len(articles),
            "ground_truth_tier": ground_truth_tier,
        },
        perspective_landscape=perspective_landscape,
        sentiment_timeline=sentiment_curve,
        inflection_point_cards=inflection_points,
        frame_evolution_log=frame_sequence,
        voice_composition_shifts=voice_shifts,
        narrative_intelligence_summary=narrative_summary,
        ground_truth_tier=ground_truth_tier,
        time_buckets=buckets,
    )

    log.info(
        "analyst complete — buckets=%d, inflections=%d, tier=%s",
        len(buckets), len(inflection_points), ground_truth_tier,
    )

    return {
        "narrative_report": report.model_dump(mode="json"),
        "analysis_complete": True,
        "worker_outputs": [
            f"[analyst]\n"
            f"Buckets: {len(buckets)} | Inflection points: {len(inflection_points)} | "
            f"Ground truth tier: {ground_truth_tier} | "
            f"Frame sequence: {' → '.join(fp.frame_type for fp in frame_sequence)}"
        ],
    }
