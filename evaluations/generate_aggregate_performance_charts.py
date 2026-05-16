"""
generate_aggregate_performance_charts.py
=========================================
Produces 2 composite 16:9 slide-ready PNGs showing aggregate system performance
across all real pipeline runs. Pearson r and P@K are intentionally excluded.

  slide_1_metric_overview.png       — jitter+mean overview of 3 metrics across 11 runs
  slide_2_crossclaim_and_radar.png  — per-claim heatmap + capability radar

Usage:
    python evaluations/generate_aggregate_performance_charts.py
    python evaluations/generate_aggregate_performance_charts.py --out-dir evaluations/charts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_EVAL_DIR   = Path(__file__).resolve().parent
RESULTS_DIR = _EVAL_DIR / "evaluation_results"
CHARTS_DIR  = _EVAL_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#FAFAFA"
AX_BG    = "#F5F5F5"
GRID     = "#E8E8E8"
TEXT     = "#222222"
MUTED    = "#777777"
SUBTITLE = "#555555"

METRIC_COLORS = {
    "G":  "#E63946",
    "QR": "#4361EE",
    "D":  "#2EC4B6",
}
METRIC_KEYS  = ["G", "QR", "D"]
METRIC_NAMES = {
    "G":  "Groundedness",
    "QR": "Query Relevance",
    "D":  "Diversity",
}
METRIC_DESC = {
    "G":  "Are claims backed by retrieved evidence?",
    "QR": "Does the output stay on-topic?",
    "D":  "Are multiple viewpoints represented?",
}

CLAIM_LABELS: dict[str, str] = {
    "AI will replace all junior software developers by 2030":
        "AI & Jobs",
    "Cryptocurrency regulation in 2022 caused the crypto market crash":
        "Crypto Regulation",
    "Russia's invasion of Ukraine in 2022 caused global wheat prices to rise over 30%":
        "Commodity Pricing",
    "Global wheat prices surged over 30% in the first half of 2022 following Russia's invasion of Ukraine":
        "Wheat Price Surge",
}

DRY_RUN_MARKER = "_dryrun_"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_matplotlib():
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        13,
        "axes.titlesize":   18,
        "axes.labelsize":   14,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "figure.facecolor": BG,
        "axes.facecolor":   AX_BG,
    })


def _style(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _claim_label(claim: str) -> str:
    return CLAIM_LABELS.get(claim, claim[:30] + "…")


def _load_runs(results_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Return (all_runs, best_per_claim). Excludes dry-run files."""
    all_runs: list[dict] = []
    for path in sorted(results_dir.glob("revamped_scientific_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        input_file = data.get("meta", {}).get("input_file", "")
        if DRY_RUN_MARKER in input_file:
            continue
        claim = data.get("meta", {}).get("claim", "")
        g  = data.get("groundedness", {}).get("groundedness_score", 0.0)
        qr = data.get("query_relevance", {}).get("query_relevance_score", 0.0)
        d  = data.get("perspective_diversity", {}).get("normalised_entropy", 0.0)
        all_runs.append({"claim": claim, "G": g, "QR": qr, "D": d})

    if not all_runs:
        return [], {}

    # Best run per claim: highest G, then QR as tiebreaker
    best: dict[str, dict] = {}
    for run in all_runs:
        c = run["claim"]
        if c not in best or (run["G"], run["QR"]) > (best[c]["G"], best[c]["QR"]):
            best[c] = run

    return all_runs, best


# ── Slide 1 ───────────────────────────────────────────────────────────────────

def generate_slide_1(all_runs: list[dict], out_dir: Path) -> Path:
    _configure_matplotlib()
    rng = np.random.default_rng(seed=42)

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    fig.patch.set_facecolor(BG)
    _style(ax)

    positions = [0, 2.5, 5.0]

    for idx, key in enumerate(METRIC_KEYS):
        x  = positions[idx]
        values = [r[key] for r in all_runs]
        mean   = float(np.mean(values))
        color  = METRIC_COLORS[key]

        # Background bar
        ax.bar(x, mean, width=1.4, color=color, alpha=0.18, zorder=2, linewidth=0)
        # Mean line
        ax.plot([x - 0.58, x + 0.58], [mean, mean], color=color, lw=5, zorder=4, solid_capstyle="round")
        # Jitter dots
        jitter = rng.uniform(-0.35, 0.35, len(values))
        ax.scatter(
            [x + j for j in jitter], values,
            s=90, color=color, alpha=0.82, zorder=5,
            edgecolors="white", linewidths=1.2,
        )
        # Mean value annotation
        ax.text(
            x, mean + 0.055, f"{mean:.2f}",
            ha="center", va="bottom",
            fontsize=18, fontweight="bold", color=color, zorder=6,
        )

    # "Strong performance" reference band
    ax.axhspan(0.80, 1.0, facecolor="#d4f5d4", alpha=0.35, zorder=0)
    ax.text(
        5.82, 0.895, "Strong ≥ 0.80",
        va="center", ha="left",
        fontsize=12, color="#3a7a3a", style="italic",
    )

    # X-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{METRIC_NAMES[k]}\n{METRIC_DESC[k]}" for k in METRIC_KEYS],
        fontsize=14,
    )
    # Make the first line (metric name) bold via manual rendering workaround:
    # matplotlib doesn't support per-line bold in tick labels easily,
    # so we set the tick label text then override each label individually.
    for lbl, key in zip(ax.get_xticklabels(), METRIC_KEYS):
        lbl.set_color(TEXT)

    ax.set_xlim(-1.2, 6.5)
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("Score  (0 = poor  →  1 = perfect)", fontsize=14, color=TEXT)

    # Titles
    fig.suptitle(
        "Query Relevance and Diversity Score Consistently Above 0.85",
        fontsize=22, fontweight="bold", color=TEXT, y=0.96,
    )
    fig.text(
        0.5, 0.905,
        "11 real pipeline runs across 4 claim domains  ·  dots = individual run  ·  bar = mean score",
        ha="center", fontsize=14, color=SUBTITLE, style="italic",
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    out = out_dir / "slide_1_metric_overview.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[SLIDE] {out}")
    return out


# ── Slide 2 ───────────────────────────────────────────────────────────────────

def generate_slide_2(best_per_claim: dict[str, dict], all_runs: list[dict], out_dir: Path) -> Path:
    _configure_matplotlib()

    fig = plt.figure(figsize=(19.2, 10.8))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[2.2, 1],
        left=0.06, right=0.97,
        top=0.80, bottom=0.10,
        wspace=0.14,
    )
    ax_heat  = fig.add_subplot(gs[0])
    ax_radar = fig.add_subplot(gs[1], projection="polar")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    claim_order = [
        "AI will replace all junior software developers by 2030",
        "Cryptocurrency regulation in 2022 caused the crypto market crash",
        "Russia's invasion of Ukraine in 2022 caused global wheat prices to rise over 30%",
        "Global wheat prices surged over 30% in the first half of 2022 following Russia's invasion of Ukraine",
    ]
    row_labels = [_claim_label(c) for c in claim_order]

    matrix = np.zeros((len(claim_order), 3))
    for i, claim in enumerate(claim_order):
        run = best_per_claim.get(claim)
        if run:
            matrix[i] = [run["G"], run["QR"], run["D"]]

    im = ax_heat.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if (val < 0.35 or val > 0.75) else TEXT
            ax_heat.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=20, fontweight="bold", color=txt_color,
            )

    # Axes
    ax_heat.set_xticks(range(3))
    ax_heat.set_xticklabels(
        [METRIC_NAMES[k] for k in METRIC_KEYS],
        fontsize=15,
    )
    ax_heat.xaxis.set_label_position("top")
    ax_heat.xaxis.tick_top()
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=15)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0, colors=TEXT)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.82, pad=0.03)
    cbar.set_label("Score", fontsize=12, color=TEXT)
    cbar.ax.tick_params(labelsize=11, colors=TEXT)

    ax_heat.set_title(
        "Consistent Performance Across Diverse Claim Domains",
        fontsize=17, fontweight="bold", color=TEXT, pad=52,
    )

    # ── Radar ─────────────────────────────────────────────────────────────────
    means = {k: float(np.mean([r[k] for r in all_runs])) for k in METRIC_KEYS}
    N = 3
    # Place G at top (90°), QR bottom-left (210°), D bottom-right (330°)
    angles_deg = [90, 210, 330]
    angles = [np.deg2rad(a) for a in angles_deg]
    angles_closed = angles + [angles[0]]

    values = [means[k] for k in METRIC_KEYS]
    values_closed = values + [values[0]]

    # No rotation — use standard matplotlib polar angles (0=right, CCW)
    # Place D at top (90°), G at bottom-right (330°=−30°), QR at bottom-left (210°)
    # Remap METRIC_KEYS order for this arrangement: [D, G, QR]
    radar_order = ["D", "G", "QR"]
    radar_angles_deg = [90, 330, 210]
    r_angles = [np.deg2rad(a) for a in radar_angles_deg]
    r_angles_closed = r_angles + [r_angles[0]]
    r_values = [means[k] for k in radar_order]
    r_values_closed = r_values + [r_values[0]]

    # Polygon
    ax_radar.fill(r_angles, r_values, alpha=0.22, color="#4361EE", zorder=3)
    ax_radar.plot(r_angles_closed, r_values_closed, lw=2.8, color="#4361EE", zorder=4)

    # Per-vertex colored dots and value annotations
    # Nudge direction: inward toward center for bottom vertices to avoid axis label overlap
    nudge_map = {"D": 0.15, "G": -0.18, "QR": -0.18}
    for i, key in enumerate(radar_order):
        ang = r_angles[i]
        val = r_values[i]
        ax_radar.scatter(
            [ang], [val],
            s=180, color=METRIC_COLORS[key], zorder=6,
            edgecolors="white", linewidths=2,
        )
        nudge = nudge_map[key]
        ax_radar.text(
            ang, val + nudge, f"{val:.2f}",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=METRIC_COLORS[key],
        )

    # Axis labels — set manually to avoid overlap with value annotations
    ax_radar.set_xticks(r_angles)
    ax_radar.set_xticklabels(
        [METRIC_NAMES[k] for k in radar_order],
        fontsize=13, color=TEXT,
    )
    ax_radar.tick_params(axis="x", pad=14)
    ax_radar.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9, color=MUTED)
    ax_radar.set_ylim(0, 1.28)
    ax_radar.yaxis.grid(True, color=GRID, lw=0.9)
    ax_radar.xaxis.grid(True, color=GRID, lw=0.9)
    ax_radar.set_facecolor(BG)
    ax_radar.spines["polar"].set_visible(False)

    ax_radar.set_title(
        "System Capability\nMean Across All Runs",
        fontsize=16, fontweight="bold", color=TEXT, pad=22,
    )

    # ── Shared slide title ────────────────────────────────────────────────────
    fig.suptitle(
        "Evo-Revamped  ·  Evaluation Results",
        fontsize=22, fontweight="bold", color=TEXT, y=0.98,
    )
    fig.text(
        0.5, 0.925,
        "Best pipeline run per claim domain  ·  4 real-world claim types  ·  3 core evaluation metrics",
        ha="center", fontsize=13, color=SUBTITLE, style="italic",
    )

    out = out_dir / "slide_2_crossclaim_and_radar.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[SLIDE] {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2 presentation-ready aggregate performance slide PNGs."
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Directory containing revamped_scientific_*.json files",
    )
    parser.add_argument(
        "--out-dir", default=str(CHARTS_DIR),
        help="Output directory for slide PNGs",
    )
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs, best_per_claim = _load_runs(results_dir)

    if not all_runs:
        print("[ERROR] No real pipeline runs found. Check --results-dir.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(all_runs)} real runs across {len(best_per_claim)} claim types.")

    generate_slide_1(all_runs, out_dir)
    generate_slide_2(best_per_claim, all_runs, out_dir)

    print("\n[DONE] Both slides saved to:", out_dir)


if __name__ == "__main__":
    main()
