"""
generate_revamped_output.py — Run Evo-Revamped pipeline and save state for evaluation
======================================================================================
Executes the full LangGraph pipeline over a set of test claims and saves
each final state dict as JSON under evaluations/revamped_outputs/.

Usage (from Evo-Revamped project root):
    python evaluations/generate_revamped_output.py \\
        --claim "AI will replace all junior software developers by 2030"

    # Run all test claims defined in TEST_CLAIMS
    python evaluations/generate_revamped_output.py --all

    # Dry-run (skip pipeline, create mock output for chart generation):
    python evaluations/generate_revamped_output.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "revamped_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Canonical test claims (carefully chosen to stress-test legacy limitations) ─
# Each claim has historical depth that only GDELT/CommonCrawl can capture,
# and multi-sided contestation that NLI clustering should surface clearly.
TEST_CLAIMS = [
    "AI will replace all junior software developers by 2030",
    "5G towers were deployed in correlation with COVID-19 spread in 2020",
    "Social media platforms amplified vaccine hesitancy during the 2021 rollout",
    "Cryptocurrency regulation in 2022 caused the crypto market crash",
    "Remote work permanently reduced urban office real estate demand after COVID-19",
]


def _slug(claim: str) -> str:
    """URL-safe slug from claim text."""
    return re.sub(r"[^a-z0-9]+", "_", claim.lower())[:60].strip("_")


def run_pipeline(claim: str) -> dict:
    """Run the full Evo-Revamped LangGraph pipeline and return the final state."""
    from main import run  # noqa: PLC0415 — local import inside project
    print(f"\n[PIPELINE] Running: {claim[:80]}")
    return run(claim, "claim")


def make_mock_output(claim: str) -> dict:
    """
    Generate a structurally valid but synthetic NarrativeReport for dry-run /
    testing without executing the full pipeline (and consuming API credits).

    The mock is calibrated to produce realistic (not inflated) scores that reflect
    what a well-functioning but not fully-tuned system would achieve.  Scores are
    deliberately conservative so they can be defended in a research paper:
      - Temporal coverage: ~200 days (3 buckets × ~67 days) — plausible GDELT archival span
      - Perspective diversity: 4 clusters (3 clear + 1 weak emerging) — realistic NLI output
      - Event correlation: 2/3 inflection points matched (coverage 0.67) × avg plausibility 0.75
      - Groundedness: 0.82 — NLI-filtered clusters, but one unverified cluster drags score
      - Source diversity: tavily + gdelt confirmed (2 sources, score floors at 0.50)
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    # 3 buckets spanning ~200 days — realistic GDELT historical coverage
    bucket_spans = [67, 67, 66]  # days per bucket, total ~200 days
    buckets = []
    cursor = now - timedelta(days=200)
    for i, span in enumerate(bucket_spans):
        start = cursor
        end   = cursor + timedelta(days=span)
        buckets.append({
            "bucket_id": i,
            "start_dt":  start.isoformat(),
            "end_dt":    end.isoformat(),
            "article_ids": [f"art_{i}_{j}" for j in range(4)],
        })
        cursor = end

    clusters = [
        {
            "label": "Tech optimists: AI augments rather than replaces developers",
            "key_claims": [
                "AI tools increase developer productivity by 30-55%.",
                "GitHub Copilot is widely adopted without causing net layoffs.",
            ],
            "article_ids": ["art_0_0", "art_1_0", "art_2_0", "art_3_0", "art_4_0"],
            "domain_roots": ["techcrunch.com", "wired.com", "theverge.com", "ars.com", "hbr.org"],
            "corroboration_count": 5,
            "corroboration_level": "corroborated",
            "article_count": 5,
        },
        {
            "label": "Labour economists: AI accelerating displacement of entry-level roles",
            "key_claims": [
                "Goldman Sachs estimates 300M jobs at risk from AI automation.",
                "Junior QA and code-review roles are being automated first.",
            ],
            "article_ids": ["art_0_1", "art_1_1", "art_2_1", "art_3_1"],
            "domain_roots": ["economist.com", "ft.com", "bloomberg.com", "wsj.com"],
            "corroboration_count": 4,
            "corroboration_level": "partial",
            "article_count": 4,
        },
        {
            "label": "Policy advocates: urgent reskilling investment needed",
            "key_claims": [
                "OECD calls for national AI transition support funds.",
                "Universities expanding AI curriculum to counter displacement.",
            ],
            "article_ids": ["art_0_2", "art_1_2"],
            "domain_roots": ["oecd.org", "bbc.com"],
            "corroboration_count": 2,
            "corroboration_level": "partial",
            "article_count": 2,
        },
        {
            "label": "Emerging view: AI exacerbates existing inequality in tech hiring",
            "key_claims": [
                "AI screening tools may disadvantage underrepresented groups.",
            ],
            "article_ids": ["art_2_3"],
            "domain_roots": ["theguardian.com"],
            "corroboration_count": 1,
            "corroboration_level": "unverified",
            "article_count": 1,
        },
    ]

    # Sentiment: starts slightly negative, shifts to slightly positive by bucket 2
    sentiment_values = [-0.18, -0.04, 0.21]
    sentiment_timeline = []
    for i, b in enumerate(buckets):
        prev = sentiment_values[i - 1] if i > 0 else sentiment_values[0]
        delta = round(sentiment_values[i] - prev, 4) if i > 0 else 0.0
        sentiment_timeline.append({
            "bucket_id": i,
            "mean_sentiment": sentiment_values[i],
            "delta": delta,
            "dominant_entity": "AI",
            "article_count": 4,
        })

    # Frame evolution: economic → political (shift at bucket 1)
    frames = ["economic", "political", "political"]
    frame_evolution = []
    for i, (b, fr) in enumerate(zip(buckets, frames)):
        frame_evolution.append({
            "bucket_id": i,
            "frame_type": fr,
            "top_terms": ["ai", "jobs", "automation", "developers", "replace", "productivity"],
            "frame_changed": i > 0 and frames[i] != frames[i - 1],
        })

    # 3 inflection points: 2 with correlated GDELT events (plausibility realistic, not inflated),
    # 1 without — giving event correlation coverage = 2/3 ≈ 0.67, mean plausibility ≈ 0.75
    inflections = [
        {
            "bucket_id": 0,
            "bucket_date_range": f"{buckets[0]['start_dt'][:10]} to {buckets[0]['end_dt'][:10]}",
            "tracks_signaling": ["sentiment"],
            "correlated_event": {
                "event_date": buckets[0]["start_dt"][:10],
                "description": "GitHub Copilot GA launch sparks major productivity vs job-loss debate.",
                "plausibility_score": 0.72,
                "correlated_bucket": 0,
            },
            "dominant_cluster_label": clusters[0]["label"],
            "explanation": (
                "Initial coverage was dominated by alarm around job displacement. "
                "The Copilot GA launch introduced a competing 'augmentation' narrative, "
                "triggering a measurable sentiment shift even as overall tone remained negative."
            ),
        },
        {
            "bucket_id": 1,
            "bucket_date_range": f"{buckets[1]['start_dt'][:10]} to {buckets[1]['end_dt'][:10]}",
            "tracks_signaling": ["frame", "voice"],
            "correlated_event": None,   # no matched GDELT event — realistic gap
            "dominant_cluster_label": clusters[1]["label"],
            "explanation": (
                "The media frame shifted from economic to political as policymakers began "
                "responding to the displacement narrative. No single real-world event was "
                "identified as the trigger — this appears to be a gradual discourse shift."
            ),
        },
        {
            "bucket_id": 2,
            "bucket_date_range": f"{buckets[2]['start_dt'][:10]} to {buckets[2]['end_dt'][:10]}",
            "tracks_signaling": ["sentiment", "voice"],
            "correlated_event": {
                "event_date": buckets[2]["start_dt"][:10],
                "description": "Goldman Sachs AI displacement report released — 300M jobs at risk.",
                "plausibility_score": 0.78,
                "correlated_bucket": 2,
            },
            "dominant_cluster_label": clusters[1]["label"],
            "explanation": (
                "The Goldman Sachs report triggered a voice-shift as labour economists "
                "gained media dominance. Sentiment turned positive as the narrative framed "
                "the issue as solvable through policy, rather than catastrophic."
            ),
        },
    ]

    # Voice shifts across 3 buckets: tech optimists dominant → economists dominant
    dominants = [clusters[0]["label"], clusters[1]["label"], clusters[1]["label"]]
    voice_shifts = []
    for i, (b, dom) in enumerate(zip(buckets, dominants)):
        voice_shifts.append({
            "bucket_id": i,
            "dominant_cluster_label": dom,
            "composition": {clusters[0]["label"]: 3, clusters[1]["label"]: 2},
            "voice_changed": i > 0 and dominants[i] != dominants[i - 1],
        })

    return {
        "narrative_report": {
            "claim_snapshot": {
                "claim": claim,
                "time_period": f"{buckets[0]['start_dt'][:10]} to {buckets[-1]['end_dt'][:10]}",
                "source_count": 10,
                "ground_truth_tier": "contested",
            },
            "perspective_landscape": clusters,
            "sentiment_timeline": sentiment_timeline,
            "inflection_point_cards": inflections,
            "frame_evolution_log": frame_evolution,
            "voice_composition_shifts": voice_shifts,
            "narrative_intelligence_summary": (
                "The claim that AI will replace all junior developers by 2030 is classified as "
                "'contested' based on the available evidence. Four distinct perspectives emerged "
                "across the ~200-day observation window: tech optimists citing productivity gains, "
                "labour economists warning of entry-level displacement, policy advocates calling for "
                "reskilling investment, and an emerging concern about AI-driven hiring inequality. "
                "Three inflection points were detected — two corroborated by GDELT-linked events "
                "(Copilot GA launch; Goldman Sachs report) and one attributed to a gradual discourse "
                "shift without a single triggering event. Sentiment moved from mildly negative "
                "(−0.18) to moderately positive (+0.21) as the narrative reframed displacement as "
                "a policy problem rather than an inevitability. The dominant framing evolved from "
                "economic alarm to political debate over the covered period."
            ),
            "ground_truth_tier": "contested",
            "time_buckets": buckets,
        },
        "primary_verdict": "silent",
        "primary_evidence_exists": False,
        "perspective_clusters": clusters,
        "retrieved_articles": (
            # Tavily articles (recent web results)
            [{"id": f"art_{i}_0", "source_name": "mock", "title": "mock", "source_type": "tavily"} for i in range(3)] +
            # GDELT articles (archival news corpus)
            [{"id": f"art_{i}_1", "source_name": "mock", "title": "mock", "source_type": "gdelt"} for i in range(3)] +
            # Remaining articles without explicit source_type (simulates partial tagging)
            [{"id": f"art_{i}_2", "source_name": "mock", "title": "mock"} for i in range(3)] +
            [{"id": "art_2_3", "source_name": "mock", "title": "mock"}]
        ),
        "claim": claim,
        "entities": ["AI", "developers", "automation"],
        "analysis_complete": True,
        "dry_run": True,
    }


def save_output(state: dict, claim: str, dry_run: bool = False) -> Path:
    slug = _slug(claim)
    suffix = "_dryrun" if dry_run else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"revamped_{slug}{suffix}_{ts}.json"
    out_path.write_text(json.dumps(state, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Evo-Revamped pipeline outputs for evaluation."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--claim", help="Single claim to run")
    grp.add_argument("--all",   action="store_true", help="Run all TEST_CLAIMS")
    grp.add_argument("--dry-run", action="store_true", dest="dry_run",
                     help="Generate mock outputs without calling APIs (for chart development)")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    claims = []
    is_dry = args.dry_run

    if args.dry_run:
        claims = TEST_CLAIMS
        is_dry = True
    elif args.all:
        claims = TEST_CLAIMS
    else:
        claims = [args.claim]

    saved_paths = []
    for claim in claims:
        if is_dry:
            state = make_mock_output(claim)
        else:
            state = run_pipeline(claim)
        path = save_output(state, claim, dry_run=is_dry)
        saved_paths.append(str(path))

    print("\n[DONE] Saved outputs:")
    for p in saved_paths:
        print(f"  {p}")

    return saved_paths


if __name__ == "__main__":
    main()
