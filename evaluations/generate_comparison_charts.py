"""
generate_comparison_charts.py — Visualisation for Evo vs Evo-Revamped
=======================================================================
Reads evaluation JSONs and produces publication-ready charts.

Usage:
    python evaluations/generate_comparison_charts.py \\
        --comparison-json evaluations/comparison_results/comparison_*.json \\
        --out-dir evaluations/charts

    # Quick mode using existing Evo results + a dry-run Revamped output:
    python evaluations/generate_comparison_charts.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output dir ────────────────────────────────────────────────────────────────
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
EVO_COLOR     = "#E05C5C"   # muted red  — legacy system
REV_COLOR     = "#3A86FF"   # vivid blue — Revamped
GRID_COLOR    = "#E8E8E8"
BG_COLOR      = "#FAFAFA"
TEXT_COLOR    = "#222222"

METRIC_LABELS = {
    "groundedness":          "Groundedness\n& Factual Fidelity",
    "temporal_coverage":     "Temporal\nCoverage Depth",
    "perspective_diversity": "Perspective\nDiversity Index",
    "event_correlation":     "Event Correlation\nAccuracy",
    "source_diversity":      "Source\nDiversity Index",
}

METRIC_ORDER = list(METRIC_LABELS.keys())


# =============================================================================
# Chart 1 — Grouped Bar: per-metric score comparison
# =============================================================================

def chart_per_metric_bars(comparison: dict, out_dir: Path):
    evo_s = comparison["per_metric_scores"]["evo"]
    rev_s = comparison["per_metric_scores"]["revamped"]
    weights = comparison["meta"]["metric_weights"]

    metrics  = METRIC_ORDER
    evo_vals = [evo_s[m] for m in metrics]
    rev_vals = [rev_s[m] for m in metrics]
    wts      = [weights[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars_evo = ax.bar(x - width / 2, evo_vals, width,
                      color=EVO_COLOR, label="Evo (Legacy)", zorder=3,
                      linewidth=0, alpha=0.9)
    bars_rev = ax.bar(x + width / 2, rev_vals, width,
                      color=REV_COLOR, label="Evo-Revamped", zorder=3,
                      linewidth=0, alpha=0.9)

    # Value labels
    for bar in bars_evo:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5,
                color=TEXT_COLOR, fontweight="bold")
    for bar in bars_rev:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5,
                color=TEXT_COLOR, fontweight="bold")

    # Weight annotation below metric label
    weight_labels = [f"{METRIC_LABELS[m]}\n(weight: {int(wts[i]*100)}%)"
                     for i, m in enumerate(metrics)]

    ax.set_xticks(x)
    ax.set_xticklabels(weight_labels, fontsize=9.5, color=TEXT_COLOR)
    ax.set_ylim(0, 1.26)
    ax.set_ylabel("Score  (0.0 = worst,  1.0 = best)", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Each metric captures a distinct analytical capability; higher bars mean better performance",
                 fontsize=8.5, color="#555555", style="italic", pad=6)
    fig.suptitle("Per-Metric Score Comparison: Evo vs Evo-Revamped",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=10, framealpha=0.0, labelcolor=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "chart1_per_metric_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 2 — Radar / Spider: holistic system capability
# =============================================================================

def chart_radar(comparison: dict, out_dir: Path):
    evo_s = comparison["per_metric_scores"]["evo"]
    rev_s = comparison["per_metric_scores"]["revamped"]
    metrics = METRIC_ORDER

    labels = [METRIC_LABELS[m].replace("\n", " ") for m in metrics]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    evo_vals = [evo_s[m] for m in metrics] + [evo_s[metrics[0]]]
    rev_vals = [rev_s[m] for m in metrics] + [rev_s[metrics[0]]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(angles, evo_vals, "o-", linewidth=2, color=EVO_COLOR, label="Evo (Legacy)")
    ax.fill(angles, evo_vals, alpha=0.18, color=EVO_COLOR)

    ax.plot(angles, rev_vals, "o-", linewidth=2, color=REV_COLOR, label="Evo-Revamped")
    ax.fill(angles, rev_vals, alpha=0.20, color=REV_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5, color=TEXT_COLOR)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=7.5, color="#888888")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    ax.spines["polar"].set_visible(False)

    fig.suptitle("System Capability Radar: Evo vs Evo-Revamped",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.text(0.5, 0.01, "Larger shaded area = stronger overall system. Each axis is a different capability scored 0–1.",
             ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10),
              fontsize=10, framealpha=0.0, labelcolor=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    out_path = out_dir / "chart2_radar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 3 — Composite Score Comparison (horizontal bar)
# =============================================================================

def chart_composite(comparison: dict, out_dir: Path):
    evo_c = comparison["composite_scores"]["evo"]
    rev_c = comparison["composite_scores"]["revamped"]
    imp   = comparison["composite_scores"]["improvement_pct"]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    systems = ["Evo (Legacy)", "Evo-Revamped"]
    scores  = [evo_c, rev_c]
    colors  = [EVO_COLOR, REV_COLOR]

    bars = ax.barh(systems, scores, color=colors, height=0.42, zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", ha="left",
                fontsize=12, fontweight="bold", color=TEXT_COLOR)

    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Composite Score  (0.0 = worst,  1.0 = best)", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Weighted average of all 5 metrics — higher score = more capable system",
                 fontsize=8.5, color="#555555", style="italic", pad=4)
    fig.suptitle(f"Composite Evaluation Score — Evo-Revamped is {imp:.1f}% better overall",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.99)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / "chart3_composite.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 4 — Sentiment Timeline (Evo-Revamped only — multi-track)
# =============================================================================

def chart_sentiment_timeline(revamped_report: dict, out_dir: Path):
    timeline = revamped_report.get("narrative_report", revamped_report).get("sentiment_timeline", [])
    inflections = revamped_report.get("narrative_report", revamped_report).get("inflection_point_cards", [])

    if not timeline:
        print("[CHART] No sentiment_timeline data — skipping chart 4")
        return None

    bucket_ids = [p["bucket_id"] for p in timeline]
    sentiments = [p["mean_sentiment"] for p in timeline]
    deltas     = [p["delta"] for p in timeline]
    inf_ids    = {ip["bucket_id"] for ip in inflections}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_COLOR)

    # Top: sentiment curve
    ax1.plot(bucket_ids, sentiments, "o-", color=REV_COLOR, linewidth=2.2,
             markersize=6, zorder=4, label="Mean Sentiment")
    ax1.axhline(0, color="#AAAAAA", linewidth=0.9, linestyle="--", zorder=2)
    ax1.fill_between(bucket_ids, sentiments, 0,
                     where=[s >= 0 for s in sentiments],
                     alpha=0.15, color=REV_COLOR, zorder=2)
    ax1.fill_between(bucket_ids, sentiments, 0,
                     where=[s < 0 for s in sentiments],
                     alpha=0.15, color=EVO_COLOR, zorder=2)

    # Mark inflection points
    for bid in inf_ids:
        if bid < len(sentiments):
            ax1.axvline(bid, color="#FFA62B", linewidth=1.5,
                        linestyle=":", zorder=3, alpha=0.8)
            ax1.scatter([bid], [sentiments[bid]], color="#FFA62B",
                        s=90, zorder=5, marker="D")

    ax1.set_ylabel("Mean Sentiment Score\n(−1 = very negative,  +1 = very positive)", color=TEXT_COLOR, fontsize=10)
    fig.suptitle("Evo-Revamped: How Public Sentiment Changed Over Time",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.99)
    ax1.set_title("Each time bucket = a period of news coverage. Orange diamonds mark major narrative turning points.",
                  fontsize=8.5, color="#555555", style="italic", pad=4)
    ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(colors=TEXT_COLOR)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    inf_patch = mpatches.Patch(color="#FFA62B", label="Inflection Point")
    ax1.legend(handles=[
        mpatches.Patch(color=REV_COLOR, label="Positive Sentiment"),
        mpatches.Patch(color=EVO_COLOR, label="Negative Sentiment"),
        inf_patch,
    ], fontsize=9, framealpha=0.0, labelcolor=TEXT_COLOR)

    # Bottom: delta bars
    bar_colors = [REV_COLOR if d >= 0 else EVO_COLOR for d in deltas]
    ax2.bar(bucket_ids, deltas, color=bar_colors, zorder=3, alpha=0.85)
    ax2.axhline(0, color="#AAAAAA", linewidth=0.8, zorder=2)
    ax2.set_ylabel("Change in Sentiment\nvs. Previous Period", color=TEXT_COLOR, fontsize=9)
    ax2.set_xlabel("Time Bucket (chronological order, left = earliest)", color=TEXT_COLOR, fontsize=10)
    ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "chart4_sentiment_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Chart 5 — Perspective Cluster Breakdown (stacked bar / donut)
# =============================================================================

def chart_perspective_clusters(revamped_report: dict, out_dir: Path):
    report   = revamped_report.get("narrative_report", revamped_report)
    clusters = report.get("perspective_landscape", [])
    if not clusters:
        print("[CHART] No clusters — skipping chart 5")
        return None

    labels = [c.get("label", f"Cluster {i}")[:45] + "…"
              if len(c.get("label", "")) > 45 else c.get("label", f"Cluster {i}")
              for i, c in enumerate(clusters)]
    counts = [c.get("article_count", c.get("corroboration_count", 1)) for c in clusters]
    levels = [c.get("corroboration_level", "unverified") for c in clusters]

    level_colors = {"corroborated": "#3A86FF", "partial": "#8BC4FF", "unverified": "#C0D6F5"}
    colors = [level_colors.get(lv, "#AAAAAA") for lv in levels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(clusters) * 0.9)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos = np.arange(len(clusters))
    bars = ax.barh(y_pos, counts, color=colors, height=0.55, zorder=3)

    for bar, count, lv in zip(bars, counts, levels):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{count} articles | {lv}", va="center", fontsize=8,
                color=TEXT_COLOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, color=TEXT_COLOR)
    ax.set_xlabel("Number of Articles Supporting This Viewpoint", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Each row is a distinct perspective identified by NLI analysis. Colour shows how strongly it is backed by independent sources.",
                 fontsize=8.5, color="#555555", style="italic", pad=4)
    fig.suptitle("Evo-Revamped: Distinct Viewpoints Found in Media Coverage",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.99)

    legend_handles = [
        mpatches.Patch(color="#3A86FF", label="Strongly corroborated — confirmed by 5+ independent sources"),
        mpatches.Patch(color="#8BC4FF", label="Partially corroborated — confirmed by 3–4 sources"),
        mpatches.Patch(color="#C0D6F5", label="Unverified — only 1–2 sources found so far"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9,
              facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="upper right")
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "chart5_perspective_clusters.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[CHART] {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================

def _load_latest(directory: Path, pattern: str) -> dict | None:
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts for Evo vs Evo-Revamped."
    )
    parser.add_argument("--comparison-json", default=None,
                        help="Path to a comparison_*.json produced by run_comparison.py")
    parser.add_argument("--revamped-report", default=None,
                        help="Path to a revamped_*.json for timeline/cluster charts")
    parser.add_argument("--out-dir", default=str(CHARTS_DIR),
                        help="Output directory for chart PNGs")
    parser.add_argument("--quick", action="store_true",
                        help="Auto-discover latest result files in evaluations/")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(__file__).parent

    comparison = None
    revamped_data = None

    if args.quick or (not args.comparison_json and not args.revamped_report):
        comparison   = _load_latest(eval_dir / "comparison_results", "comparison_*.json")
        # prefer dry-run files; fall back to any revamped output
        revamped_data = (
            _load_latest(eval_dir / "revamped_outputs", "*_dryrun_*.json")
            or _load_latest(eval_dir / "revamped_outputs", "revamped_*.json")
        )
    else:
        if args.comparison_json:
            with open(args.comparison_json, encoding="utf-8") as f:
                comparison = json.load(f)
        if args.revamped_report:
            with open(args.revamped_report, encoding="utf-8") as f:
                revamped_data = json.load(f)

    generated = []
    if comparison:
        generated.append(chart_per_metric_bars(comparison, out_dir))
        generated.append(chart_radar(comparison, out_dir))
        generated.append(chart_composite(comparison, out_dir))
    else:
        print("[WARN] No comparison JSON found — skipping charts 1-3. "
              "Run run_comparison.py first, or use --quick after a dry-run.")

    if revamped_data:
        generated.append(chart_sentiment_timeline(revamped_data, out_dir))
        generated.append(chart_perspective_clusters(revamped_data, out_dir))
    else:
        print("[WARN] No Revamped report found — skipping charts 4-5. "
              "Run generate_revamped_output.py --dry-run first.")

    print(f"\n[DONE] Generated {sum(1 for g in generated if g)} chart(s) in {out_dir}/")


if __name__ == "__main__":
    main()
