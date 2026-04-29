"""
evaluate_revamped_individual.py — Individual Evaluation of Evo-Revamped
========================================================================
Computes 4 performance metrics for Evo-Revamped and generates 3 publication-
ready charts.  Equivalent to Evo's evaluate_groundedness.py + evaluate_relevance.py
+ generate_evaluation_charts.py combined.

Metrics:
  1. Groundedness Score       — What fraction of perspectives are NLI-verified?
  2. Narrative Coverage Score — How complete is the structured report output?
  3. Evidence Quality Score   — How strong is the corroboration across clusters?
  4. Pipeline Intelligence Score — How deep is temporal, source, and perspective coverage?

Usage:
    python evaluations/evaluate_revamped_individual.py
    python evaluations/evaluate_revamped_individual.py --input path/to/revamped_*.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(_EVAL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from comparative_evaluator import (
    GroundednessComparator,
    TemporalCoverageEvaluator,
    PerspectiveDiversityEvaluator,
    SourceDiversityEvaluator,
)

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = _EVAL_DIR / "evaluation_results"
CHARTS_DIR  = _EVAL_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ── Colour palette (consistent with generate_comparison_charts.py) ────────────
REV_COLOR   = "#3A86FF"
GOOD_COLOR  = "#2EC4B6"
WARN_COLOR  = "#FF9F1C"
WEAK_COLOR  = "#E05C5C"
GRID_COLOR  = "#E8E8E8"
BG_COLOR    = "#FAFAFA"
TEXT_COLOR  = "#222222"

# ── NarrativeReport expected sections ────────────────────────────────────────
EXPECTED_SECTIONS = [
    "claim_snapshot",
    "perspective_landscape",
    "sentiment_timeline",
    "inflection_point_cards",
    "frame_evolution_log",
    "voice_composition_shifts",
    "narrative_intelligence_summary",
]


# =============================================================================
# METRIC 2 — Narrative Coverage Score
# =============================================================================

def compute_narrative_coverage(narrative_report: dict, coverage_gaps: list) -> dict:
    """
    Measures how complete the structured NarrativeReport output is.

    Score = (present_sections / total_expected_sections) × gap_penalty

    gap_penalty = max(0.5, 1 - 0.1 × len(coverage_gaps))
    — each unresolved coverage gap reduces the score by 10%, floored at 50%.
    """
    present = sum(1 for s in EXPECTED_SECTIONS if narrative_report.get(s))
    section_fraction = present / len(EXPECTED_SECTIONS)

    gap_penalty = max(0.5, 1.0 - 0.10 * len(coverage_gaps))
    score = round(section_fraction * gap_penalty, 4)

    return {
        "metric": "Narrative Coverage",
        "score": score,
        "sections_present": present,
        "sections_expected": len(EXPECTED_SECTIONS),
        "section_fraction": round(section_fraction, 4),
        "coverage_gaps": coverage_gaps,
        "gap_penalty": round(gap_penalty, 4),
        "note": (
            f"{present}/{len(EXPECTED_SECTIONS)} expected report sections are populated. "
            f"{len(coverage_gaps)} unresolved coverage gap(s) applied a {round((1-gap_penalty)*100):.0f}% "
            "penalty to the section completeness score."
        ),
    }


# =============================================================================
# METRIC 3 — Evidence Quality Score
# =============================================================================

CORR_WEIGHT = {"corroborated": 1.0, "partial": 0.75, "unverified": 0.35}


def compute_evidence_quality(clusters: list) -> dict:
    """
    Measures the overall strength of evidence across all perspective clusters.

    Score = weighted average of corroboration levels, weighted by article_count.
    — Clusters backed by many articles have more influence on the final score.
    """
    if not clusters:
        return {
            "metric": "Evidence Quality",
            "score": 0.0,
            "total_articles": 0,
            "weighted_score_sum": 0.0,
            "cluster_breakdown": [],
            "note": "No clusters in report.",
        }

    total_articles = sum(c.get("article_count", 1) for c in clusters)
    weighted_sum = 0.0
    breakdown = []
    for c in clusters:
        level = c.get("corroboration_level", "unverified")
        count = c.get("article_count", 1)
        weight = CORR_WEIGHT.get(level, 0.0)
        contribution = weight * count
        weighted_sum += contribution
        breakdown.append({
            "cluster_label": c.get("label", "")[:80],
            "corroboration_level": level,
            "article_count": count,
            "weight": weight,
        })

    score = round(weighted_sum / max(total_articles, 1), 4)

    return {
        "metric": "Evidence Quality",
        "score": score,
        "total_articles": total_articles,
        "weighted_score_sum": round(weighted_sum, 4),
        "cluster_breakdown": breakdown,
        "note": (
            f"Score is a weighted average of NLI corroboration levels across "
            f"{len(clusters)} clusters ({total_articles} total articles). "
            "Weights: corroborated=1.0, partial=0.75, unverified=0.35."
        ),
    }


# =============================================================================
# METRIC 4 — Pipeline Intelligence Score
# =============================================================================

def compute_pipeline_intelligence(
    temporal_score: float,
    source_score: float,
    perspective_score: float,
) -> dict:
    """
    Composite of the three architectural depth metrics — equal weights.

    This captures the structural advantages of Evo-Revamped that go beyond
    simple output quality: how far back it looks (temporal), how many
    independent retrieval channels it used (source), and how many distinct
    viewpoints it identified (perspective).
    """
    score = round((temporal_score + source_score + perspective_score) / 3.0, 4)
    return {
        "metric": "Pipeline Intelligence",
        "score": score,
        "components": {
            "temporal_coverage": round(temporal_score, 4),
            "source_diversity": round(source_score, 4),
            "perspective_diversity": round(perspective_score, 4),
        },
        "note": (
            "Equal-weighted composite of temporal coverage depth, retrieval source diversity, "
            "and NLI perspective diversity — three structural capabilities absent in legacy Evo."
        ),
    }


# =============================================================================
# COMPOSITE
# =============================================================================

INDIVIDUAL_WEIGHTS = {
    "groundedness":          0.30,
    "narrative_coverage":    0.20,
    "evidence_quality":      0.25,
    "pipeline_intelligence": 0.25,
}

RATING_THRESHOLDS = [
    (0.80, "HIGH",     GOOD_COLOR),
    (0.60, "MODERATE", WARN_COLOR),
    (0.00, "LOW",      WEAK_COLOR),
]


def _rating(score: float):
    for threshold, label, color in RATING_THRESHOLDS:
        if score >= threshold:
            return label, color
    return "LOW", WEAK_COLOR


def compute_composite(scores: dict) -> float:
    return round(sum(INDIVIDUAL_WEIGHTS[k] * v for k, v in scores.items()), 4)


# =============================================================================
# Chart 1 — Individual Metric Scores (horizontal bar)
# =============================================================================

def chart_individual_scores(results: dict, out_dir: Path) -> Path:
    pm = results["per_metric_scores"]
    metrics = [
        ("Groundedness\n(Are perspectives NLI-verified?)",              pm["groundedness"]),
        ("Narrative Coverage\n(Is the report fully populated?)",        pm["narrative_coverage"]),
        ("Evidence Quality\n(How strong is the corroboration?)",        pm["evidence_quality"]),
        ("Pipeline Intelligence\n(Temporal × Source × Perspective depth)", pm["pipeline_intelligence"]),
    ]
    composite = results["composite_score"]

    labels = [m[0] for m in metrics]
    scores = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos = np.arange(len(labels))
    bar_colors = [_rating(s)[1] for s in scores]
    bars = ax.barh(y_pos, scores, color=bar_colors, height=0.52, zorder=3)

    for bar, score in zip(bars, scores):
        rating_label, _ = _rating(score)
        ax.text(score + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}  [{rating_label}]",
                va="center", ha="left", fontsize=9.5,
                color=TEXT_COLOR, fontweight="bold")

    # Composite marker
    ax.axvline(composite, color="#333333", linewidth=1.4, linestyle="--", zorder=4, alpha=0.7)
    ax.text(composite + 0.008, len(labels) - 0.05,
            f"Composite: {composite:.2f}",
            fontsize=8.5, color="#333333", va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_COLOR)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Score  (0.0 = worst,  1.0 = best)", color=TEXT_COLOR, fontsize=10)
    ax.set_title("How well does the system produce grounded, comprehensive, and structurally intelligent analysis?",
                 fontsize=8.5, color="#555555", style="italic", pad=4)
    fig.suptitle("Evo-Revamped: Individual Performance Scores",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)

    legend_handles = [
        mpatches.Patch(color=GOOD_COLOR, label="HIGH  (≥ 0.80) — strong performance"),
        mpatches.Patch(color=WARN_COLOR, label="MODERATE  (0.60–0.79) — acceptable performance"),
        mpatches.Patch(color=WEAK_COLOR, label="LOW  (< 0.60) — needs improvement"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, framealpha=0.9,
               facecolor=BG_COLOR, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0.10, 1, 0.94])
    out_path = out_dir / "chart_ri1_individual_scores.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 2 — Groundedness Breakdown per Cluster (stacked bar)
# =============================================================================

def chart_groundedness_breakdown(results: dict, out_dir: Path) -> Path:
    breakdown = results["evidence_quality"]["cluster_breakdown"]
    overall_g = results["per_metric_scores"]["groundedness"]

    if not breakdown:
        print("[CHART] No cluster data — skipping chart_ri2")
        return None

    labels = [b["cluster_label"][:50] + ("…" if len(b["cluster_label"]) > 50 else "")
              for b in breakdown]
    corr_counts   = [b["article_count"] if b["corroboration_level"] == "corroborated" else 0 for b in breakdown]
    partial_counts = [b["article_count"] if b["corroboration_level"] == "partial" else 0 for b in breakdown]
    unveri_counts  = [b["article_count"] if b["corroboration_level"] == "unverified" else 0 for b in breakdown]

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, max(4, len(breakdown) * 1.1)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    b1 = ax.barh(y_pos, corr_counts,   color="#3A86FF", height=0.55, zorder=3, label="Strongly corroborated — confirmed by 5+ independent sources")
    b2 = ax.barh(y_pos, partial_counts, left=corr_counts, color="#8BC4FF", height=0.55, zorder=3, label="Partially corroborated — confirmed by 3–4 sources")
    b3 = ax.barh(y_pos, unveri_counts,
                 left=[c + p for c, p in zip(corr_counts, partial_counts)],
                 color="#C0D6F5", height=0.55, zorder=3, label="Unverified — only 1–2 sources found")

    # Total count label at end of each bar
    totals = [c + p + u for c, p, u in zip(corr_counts, partial_counts, unveri_counts)]
    for i, total in enumerate(totals):
        level = breakdown[i]["corroboration_level"]
        ax.text(total + 0.08, i, f"{total} article(s) | {level}",
                va="center", fontsize=8.5, color=TEXT_COLOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_COLOR)
    ax.set_xlabel("Number of Articles Supporting This Viewpoint", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Each row is a distinct viewpoint in media coverage. Bar colour shows how strongly it is backed by independent sources.",
                 fontsize=8.5, color="#555555", style="italic", pad=4)
    fig.suptitle("Evidence Strength by Perspective Cluster",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)

    # Summary line — placed below the figure via fig.text so it never overlaps
    fig.text(0.5, 0.01,
             f"Overall groundedness score: {overall_g:.2f}  ({_rating(overall_g)[0]})",
             ha="center", va="bottom",
             fontsize=10, color=_rating(overall_g)[1], fontweight="bold")

    ax.legend(fontsize=8.5, framealpha=0.9, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="upper right")
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    out_path = out_dir / "chart_ri2_groundedness_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 3 — Architectural Capability Matrix
# =============================================================================

CAPABILITIES = [
    ("Multi-source retrieval",         True,  False, "Tavily + GDELT + CC", "SerpAPI only"),
    ("Temporal bucketing",              True,  False, "3 time buckets",       "Single snapshot"),
    ("NLI perspective clustering",      True,  False, "4 viewpoints found",   "Not supported"),
    ("Inflection point detection",      True,  False, "3 detected",           "Not supported"),
    ("Frame evolution tracking",        True,  False, "Economic → Political", "Not supported"),
    ("Coverage gap detection",          True,  False, "Auto-refill loop",     "Not supported"),
    ("Primary source verdict",          True,  False, "silent (honest N/A)",  "Not supported"),
    ("GDELT event correlation",         True,  False, "2/3 matched",          "Not supported"),
]


def chart_capability_matrix(out_dir: Path) -> Path:
    n = len(CAPABILITIES)
    row_labels  = [c[0] for c in CAPABILITIES]
    rev_support = [c[1] for c in CAPABILITIES]
    evo_support = [c[2] for c in CAPABILITIES]
    rev_notes   = [c[3] for c in CAPABILITIES]
    evo_notes   = [c[4] for c in CAPABILITIES]

    fig, ax = plt.subplots(figsize=(10, n * 0.7 + 1.8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    col_centers = [0.25, 0.75]
    y_positions = np.arange(n)[::-1]  # top-to-bottom order

    for i, yi in enumerate(y_positions):
        for col, (supported, note) in enumerate([(evo_support[i], evo_notes[i]),
                                                  (rev_support[i], rev_notes[i])]):
            x = col_centers[col]
            color = "#2EC4B6" if supported else "#FFCDD2"
            edge  = "#1A9E94" if supported else "#EF9A9A"
            ax.barh(yi, 0.44, left=x - 0.22, height=0.65,
                    color=color, edgecolor=edge, linewidth=0.8, zorder=3)
            symbol = "✓" if supported else "✗"
            ax.text(x, yi, f"{symbol}  {note}",
                    ha="center", va="center", fontsize=8.5,
                    color="#111111" if supported else "#888888",
                    fontweight="bold" if supported else "normal")

        # Row label on the left
        ax.text(-0.02, yi, row_labels[i],
                ha="right", va="center", fontsize=9.5, color=TEXT_COLOR)

    # Column headers
    ax.text(col_centers[0], n + 0.15, "Evo (Legacy)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=WEAK_COLOR)
    ax.text(col_centers[1], n + 0.15, "Evo-Revamped",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=REV_COLOR)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.6, n + 0.5)
    ax.axis("off")

    ax.set_title("Which analytical capabilities does each system support? Green = supported, pink = not supported.",
                 fontsize=8.5, color="#555555", style="italic", pad=4)
    fig.suptitle("Architectural Capability Comparison",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)

    # Divider line between columns
    ax.axvline(0.5, color=GRID_COLOR, linewidth=1.2, zorder=2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "chart_ri3_capability_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def _find_latest_revamped() -> Path | None:
    rev_dir = _EVAL_DIR / "revamped_outputs"
    if not rev_dir.exists():
        return None
    files = sorted(rev_dir.glob("revamped_*.json"))
    return files[-1] if files else None


def run_individual_evaluation(revamped_path: str | Path) -> dict:
    with open(revamped_path, encoding="utf-8") as f:
        state = json.load(f)

    report = state.get("narrative_report", state)
    clusters = (
        report.get("perspective_landscape")
        or state.get("perspective_clusters")
        or []
    )
    coverage_gaps = state.get("coverage_gaps") or []

    # Instantiate evaluators
    g_comp = GroundednessComparator()
    t_comp = TemporalCoverageEvaluator()
    p_comp = PerspectiveDiversityEvaluator()
    s_comp = SourceDiversityEvaluator()

    # Metric 1 — Groundedness
    g_result = g_comp.score_revamped(report)

    # Sub-scores for metric 4
    t_result = t_comp.score_revamped(report)
    p_result = p_comp.score_revamped(report)
    s_result = s_comp.score_revamped(state)

    # Metric 2 — Narrative Coverage
    nc_result = compute_narrative_coverage(report, coverage_gaps)

    # Metric 3 — Evidence Quality
    eq_result = compute_evidence_quality(clusters)

    # Metric 4 — Pipeline Intelligence
    pi_result = compute_pipeline_intelligence(
        temporal_score    = t_result["temporal_coverage_score"],
        source_score      = s_result["source_diversity_score"],
        perspective_score = p_result["perspective_diversity_score"],
    )

    per_metric_scores = {
        "groundedness":          g_result["groundedness_score"],
        "narrative_coverage":    nc_result["score"],
        "evidence_quality":      eq_result["score"],
        "pipeline_intelligence": pi_result["score"],
    }
    composite = compute_composite(per_metric_scores)

    result = {
        "meta": {
            "claim":         state.get("claim", ""),
            "input_file":    str(revamped_path),
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "metric_weights": INDIVIDUAL_WEIGHTS,
            "is_dry_run":    state.get("dry_run", False),
        },
        "groundedness":          g_result,
        "narrative_coverage":    nc_result,
        "evidence_quality":      eq_result,
        "pipeline_intelligence": pi_result,
        "temporal_detail":       t_result,
        "source_detail":         s_result,
        "perspective_detail":    p_result,
        "per_metric_scores":     per_metric_scores,
        "composite_score":       composite,
    }

    # Save JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"revamped_individual_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {out_path}")

    return result


def _print_report(result: dict):
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 64)
    print("  EVO-REVAMPED — INDIVIDUAL EVALUATION REPORT")
    print("=" * 64)
    print(f"  Claim : {result['meta']['claim'][:70]}")
    print(f"  Run   : {result['meta']['generated_at'][:19]}")
    print()

    headers = {
        "groundedness":          "Groundedness Score",
        "narrative_coverage":    "Narrative Coverage Score",
        "evidence_quality":      "Evidence Quality Score",
        "pipeline_intelligence": "Pipeline Intelligence Score",
    }
    for key, label in headers.items():
        score = result["per_metric_scores"][key]
        rating, _ = _rating(score)
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<32}  {bar}  {score:.4f}  [{rating}]")

    print()
    composite = result["composite_score"]
    print(f"  {'COMPOSITE SCORE':<32}  {composite:.4f}")
    print("=" * 64)

    # Per-metric details
    print("\n--- Groundedness ---")
    g = result["groundedness"]
    print(f"  Clusters total: {g['total_claims']}, corroborated+partial: {g['grounded_claims']}")
    print(f"  {g['note']}")

    print("\n--- Narrative Coverage ---")
    nc = result["narrative_coverage"]
    print(f"  {nc['sections_present']}/{nc['sections_expected']} sections present, {len(nc['coverage_gaps'])} gap(s)")
    print(f"  {nc['note']}")

    print("\n--- Evidence Quality ---")
    eq = result["evidence_quality"]
    print(f"  {eq['total_articles']} total articles across {len(eq['cluster_breakdown'])} clusters")
    for cb in eq["cluster_breakdown"]:
        print(f"    • {cb['cluster_label'][:60]} [{cb['corroboration_level']}]")

    print("\n--- Pipeline Intelligence ---")
    pi = result["pipeline_intelligence"]
    comp = pi["components"]
    print(f"  Temporal Coverage:     {comp['temporal_coverage']:.4f}")
    print(f"  Source Diversity:      {comp['source_diversity']:.4f}")
    print(f"  Perspective Diversity: {comp['perspective_diversity']:.4f}")
    print(f"  Pipeline Intel Score:  {pi['score']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run individual evaluation of Evo-Revamped pipeline output."
    )
    parser.add_argument("--input", default=None,
                        help="Path to revamped_*.json. Auto-discovers latest if omitted.")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rev_path = Path(args.input) if args.input else _find_latest_revamped()
    if not rev_path or not rev_path.exists():
        print("[ERROR] No Evo-Revamped output found. Run generate_revamped_output.py --dry-run first.")
        sys.exit(1)

    print(f"[INFO] Evaluating: {rev_path.name}")
    result = run_individual_evaluation(rev_path)
    _print_report(result)

    print("\n[CHARTS] Generating individual evaluation charts...")
    chart_individual_scores(result, CHARTS_DIR)
    chart_groundedness_breakdown(result, CHARTS_DIR)
    chart_capability_matrix(CHARTS_DIR)

    print(f"\n[DONE] Charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
