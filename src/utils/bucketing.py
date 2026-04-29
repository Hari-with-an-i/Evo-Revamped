"""
Adaptive time bucketing for the narrative analysis pipeline.

Groups articles into N equal-width time intervals where:
  N = max(min_buckets, article_count // 5), capped at article_count.

Articles with identical timestamps or a single-article corpus collapse into 1 bucket.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.schemas.narrative_report import TimeBucket


def build_time_buckets(
    articles: list[dict],
    min_buckets: int = 3,
    predefined_periods: list[dict] | None = None,
) -> tuple[list[TimeBucket], list[int]]:
    """
    Partition articles into adaptive time buckets or predefined periods.

    Returns:
        (buckets, assignments) where assignments[i] = bucket_id for articles[i].
    """
    if not articles:
        return [], []

    from src.workers._normalization import parse_date  # late import avoids circular dependency

    raw_dates: list[datetime | None] = []
    for a in articles:
        raw = a.get("published_at")
        dt = parse_date(raw)
        if dt is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        raw_dates.append(dt)

    known = [d for d in raw_dates if d is not None]
    if not known:
        # No articles have parseable dates — single bucket
        now = datetime.now(timezone.utc)
        bucket = TimeBucket(
            bucket_id=0,
            start_dt=now,
            end_dt=now,
            article_ids=[a.get("id", "") for a in articles],
        )
        return [bucket], [0] * len(articles)

    corpus_start = min(known)
    corpus_end = max(known)
    dates: list[datetime] = [d if d is not None else corpus_start for d in raw_dates]

    if predefined_periods:
        # Use predefined periods as buckets
        buckets = []
        for i, period in enumerate(predefined_periods):
            start = parse_date(period.get("start_date"))
            end = parse_date(period.get("end_date"))
            # Fallback if parsing fails
            if not start: start = corpus_start
            if not end: end = corpus_end
            if start.tzinfo is None: start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None: end = end.replace(tzinfo=timezone.utc)
            
            # Ensure end is strictly after start, else pad it slightly so it spans
            if end <= start:
                from datetime import timedelta
                end = start + timedelta(days=1)

            buckets.append(TimeBucket(
                bucket_id=i,
                start_dt=start,
                end_dt=end,
                article_ids=[],
            ))

        assignments = []
        for i, dt in enumerate(dates):
            # Find the best bucket for this date
            assigned = False
            for b_idx, b in enumerate(buckets):
                if b.start_dt <= dt <= b.end_dt:
                    assignments.append(b_idx)
                    buckets[b_idx].article_ids.append(articles[i].get("id", ""))
                    assigned = True
                    break
            
            # If it didn't fit neatly into any bucket, put it in the closest one
            if not assigned:
                closest_b_idx = 0
                min_dist = float('inf')
                for b_idx, b in enumerate(buckets):
                    dist_start = abs((dt - b.start_dt).total_seconds())
                    dist_end = abs((dt - b.end_dt).total_seconds())
                    dist = min(dist_start, dist_end)
                    if dist < min_dist:
                        min_dist = dist
                        closest_b_idx = b_idx
                assignments.append(closest_b_idx)
                buckets[closest_b_idx].article_ids.append(articles[i].get("id", ""))
                
        return buckets, assignments

    # All same timestamp → single bucket
    if corpus_start == corpus_end:
        bucket = TimeBucket(
            bucket_id=0,
            start_dt=corpus_start,
            end_dt=corpus_end,
            article_ids=[a.get("id", "") for a in articles],
        )
        return [bucket], [0] * len(articles)

    n = max(min_buckets, len(articles) // 5)
    n = min(n, len(articles))  # never more buckets than articles

    span = corpus_end - corpus_start
    bucket_width = span / n

    # Assign each article to a bucket
    assignments = []
    for dt in dates:
        idx = int((dt - corpus_start) / bucket_width)
        idx = min(idx, n - 1)  # clamp edge case where dt == corpus_end
        assignments.append(idx)

    # Build bucket article_id lists
    bucket_article_ids: list[list[str]] = [[] for _ in range(n)]
    for i, bucket_id in enumerate(assignments):
        art_id = articles[i].get("id", "")
        bucket_article_ids[bucket_id].append(art_id)

    buckets = []
    for b in range(n):
        b_start = corpus_start + bucket_width * b
        b_end = corpus_start + bucket_width * (b + 1)
        buckets.append(TimeBucket(
            bucket_id=b,
            start_dt=b_start,
            end_dt=b_end,
            article_ids=bucket_article_ids[b],
        ))

    return buckets, assignments
