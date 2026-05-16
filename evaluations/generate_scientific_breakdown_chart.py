"""
generate_scientific_breakdown_chart.py
=======================================
Produces 5 individual publication-ready PNGs — one per scientific metric.

  chart_sci1_groundedness.png       — per-claim NLI vs semantic scores
  chart_sci2_query_relevance.png    — 3 sub-dimensions as grouped bars
  chart_sci3_perspective_diversity.png — cluster distribution + Shannon H
  chart_sci4_sentiment_trend.png    — scatter + regression + Pearson r
  chart_sci5_event_correlation.png  — timeline dot-plot with GDELT callouts

Usage:
    python evaluations/generate_scientific_breakdown_chart.py
    python evaluations/generate_scientific_breakdown_chart.py --input path/to/revamped_scientific_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

_EVAL_DIR   = Path(__file__).resolve().parent
RESULTS_DIR = _EVAL_DIR / "evaluation_results"
CHARTS_DIR  = _EVAL_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#FAFAFA"
GRID     = "#E8E8E8"
TEXT     = "#222222"
MUTED    = "#777777"
BLUE     = "#3A86FF"
TEAL     = "#2EC4B6"
ORANGE   = "#FF9F1C"
RED      = "#E05C5C"
LIGHT    = "#C0D6F5"
MID_BLUE = "#8BC4FF"
SUBTITLE = "#555555"


def _style(ax):
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _titles(fig, ax, main: str, sub: str):
    """Suptitle + italic subtitle that never collide."""
    fig.suptitle(main, fontsize=14, fontweight="bold", color=TEXT, y=0.98)
    ax.set_title(sub, fontsize=9, color=SUBTITLE, style="italic", pad=6)


def _find_latest_scientific() -> Path | None:
    files = sorted(RESULTS_DIR.glob("revamped_scientific_*.json"))
    return files[-1] if files else None


# =============================================================================
# Chart 1 — Groundedness
# =============================================================================
def chart_groundedness(data: dict, out_dir: Path) -> Path:
    claims   = data["claim_details"]
    g_score  = data["groundedness_score"]
    grounded = data["grounded_claims"]
    total    = data["total_claims"]

    nli  = [c["nli_entailment_score"] for c in claims]
    sim  = [c["semantic_similarity"]   for c in claims]
    ok   = [c["is_grounded"]           for c in claims]
    n    = len(claims)

    SPACING = 1.8
    BAR_H   = 0.55
    NLI_THRESHOLD = 0.30

    y = np.arange(n) * SPACING

    fig_h = max(10, n * SPACING * 0.85 + 5)
    fig, ax = plt.subplots(figsize=(20, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Alternating row stripes
    for i in range(n):
        stripe = "#ECECEC" if i % 2 == 0 else BG
        ax.axhspan(y[i] - SPACING / 2 + 0.06,
                   y[i] + SPACING / 2 - 0.06,
                   facecolor=stripe, alpha=1.0, zorder=0)

    for i, (ni, si, gi) in enumerate(zip(nli, sim, ok)):
        c_main = TEAL if gi else RED
        c_edge = "#1A9E94" if gi else "#C03030"

        # Primary bar: SBERT similarity
        ax.barh(y[i], si, height=BAR_H, color=c_main, alpha=0.82,
                edgecolor=c_edge, linewidth=1.0, zorder=3)

        # SBERT label: inside bar if wide enough, else outside to the right
        if si >= 0.70:
            ax.text(si - 0.025, y[i], f"SBERT: {si:.3f}",
                    va="center", ha="right", fontsize=12,
                    color="white", fontweight="bold", zorder=6)
        else:
            ax.text(si + 0.025, y[i], f"SBERT: {si:.3f}",
                    va="center", ha="left", fontsize=12,
                    color=TEXT, fontweight="bold", zorder=6)

        # NLI diamond marker — filled if ≥ threshold, hollow if below
        nli_face  = c_edge if ni >= NLI_THRESHOLD else "white"
        nli_color = c_edge if ni >= NLI_THRESHOLD else MUTED
        ax.scatter(ni, y[i], marker="D", s=160,
                   facecolors=nli_face, edgecolors=nli_color,
                   linewidths=2.0, zorder=7, clip_on=False)

        # NLI label: de-emphasised below bar when negligible, bold above when meaningful
        NLI_LABEL_OFFSET_Y = BAR_H / 2 + 0.14
        if ni < 0.05:
            ax.text(max(ni, 0.015), y[i] - NLI_LABEL_OFFSET_Y,
                    f"NLI: {ni:.3f}", va="top", ha="left",
                    fontsize=9, color=MUTED, style="italic", zorder=6)
        else:
            ax.text(ni, y[i] + NLI_LABEL_OFFSET_Y,
                    f"NLI: {ni:.3f}", va="bottom", ha="center",
                    fontsize=11, color=nli_color, fontweight="bold", zorder=6)

        # Status badge — fixed far-right, boxed
        badge = "✓  GROUNDED" if gi else "✗  UNGROUNDED"
        ax.text(1.08, y[i], badge,
                va="center", ha="left", fontsize=13, fontweight="bold",
                color=c_edge,
                bbox=dict(boxstyle="round,pad=0.40", facecolor=c_main,
                          edgecolor=c_edge, alpha=0.18, linewidth=1.5),
                zorder=5)

    # Threshold line — label pinned at top via axes transform (never overlaps bars)
    ax.axvline(NLI_THRESHOLD, color="#888888", linewidth=1.8,
               linestyle="--", zorder=4, alpha=0.85)
    ax.text(NLI_THRESHOLD + 0.01, 0.99, "threshold = 0.30",
            transform=ax.get_xaxis_transform(),
            fontsize=10, color=MUTED, va="top", ha="left", fontweight="bold")

    # Y-axis: full claim text, up to 3 lines
    wrapped_labels = []
    for c in claims:
        lines = textwrap.fill(c["claim"], width=52).split("\n")
        lbl = "\n".join(lines[:3])
        if len(lines) > 3:
            lbl += "…"
        wrapped_labels.append(lbl)

    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_labels, fontsize=13, color=TEXT, linespacing=1.45)
    ax.tick_params(axis="y", length=0, pad=14)

    ax.set_ylim(y[0] - SPACING / 2 - 0.20, y[-1] + SPACING / 2 + 0.40)
    ax.set_xlim(0, 1.55)
    ax.set_xlabel("Score  (0.0 = no match  →  1.0 = perfect match)",
                  fontsize=15, color=TEXT, labelpad=14)

    _style(ax)
    ax.yaxis.grid(False)

    # Legend
    sbert_g    = mpatches.Patch(facecolor=TEAL, alpha=0.82,
                                label="SBERT Similarity — grounded claim (bar)")
    sbert_u    = mpatches.Patch(facecolor=RED,  alpha=0.82,
                                label="SBERT Similarity — ungrounded claim (bar)")
    nli_strong = Line2D([0], [0], marker="D", color="w",
                        markerfacecolor=TEAL, markeredgecolor="#1A9E94",
                        markersize=10,
                        label="NLI Entailment ≥ 0.30 — logically supports claim (filled ◆)")
    nli_weak   = Line2D([0], [0], marker="D", color="w",
                        markerfacecolor="white", markeredgecolor=MUTED,
                        markersize=10,
                        label="NLI Entailment < 0.30 — weak logical support (hollow ◇)")
    fig.legend(handles=[sbert_g, sbert_u, nli_strong, nli_weak],
               fontsize=12, framealpha=0.97,
               facecolor=BG, edgecolor=GRID,
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0),
               borderpad=1.0, labelspacing=0.9)

    # Title block — three separate lines, no collision
    fig.suptitle("Groundedness Evaluation — Per-Claim Signal Breakdown",
                 fontsize=20, fontweight="bold", color=TEXT, y=0.99)
    fig.text(0.5, 0.96,
             f"Overall groundedness score: {g_score:.2f}   |   "
             f"{grounded} of {total} claims grounded",
             ha="center", fontsize=15, color=SUBTITLE)
    fig.text(0.5, 0.93,
             "Bar = SBERT cosine similarity  ·  Diamond = NLI entailment  ·  "
             "Claim grounded if either signal ≥ 0.30",
             ha="center", fontsize=12, color=MUTED, style="italic")

    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    out = out_dir / "chart_sci1_groundedness.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] {out}")
    return out


# =============================================================================
# Chart 2 — Query Relevance
# =============================================================================
def chart_query_relevance(data: dict, out_dir: Path) -> Path:
    sem    = data["semantic_relevance"]
    cov    = data["coverage"]["coverage_score"]
    spec   = data["specificity"]["specificity_score"]
    on_t   = data["specificity"]["on_topic_similarity"]
    off_t  = data["specificity"]["off_topic_similarity"]
    total  = data["query_relevance_score"]
    terms  = data["coverage"]["terms"]
    covered = data["coverage"]["covered_terms"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    # ── Sub-plot A: All dimensions bar chart ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    dim_labels = ["Semantic\n(summary)", "Semantic\n(clusters)",
                  "Coverage\n(term match)", "Specificity\n(on/off topic)", "Final\nScore"]
    values     = [sem["narrative_summary"], sem["perspective_clusters"],
                  cov, spec, total]
    colors     = [BLUE, MID_BLUE, ORANGE, TEAL, "#333333"]
    weights    = ["35%", "35%", "40%", "25%", "weighted\ncomposite"]
    x = np.arange(len(dim_labels))
    bars = ax.bar(x, values, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                color=TEXT, fontweight="bold")
    for xi, w in zip(x, weights):
        ax.text(xi, -0.11, w, ha="center", fontsize=7.5, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=8.5, color=TEXT)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score (0–1)", fontsize=9, color=TEXT)
    ax.set_title("Sub-dimension Scores", fontsize=10,
                 fontweight="bold", color=TEXT, pad=4)
    _style(ax)

    # ── Sub-plot B: Coverage term chart ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    term_colors = [TEAL if t in covered else RED for t in terms]
    ax2.barh(np.arange(len(terms)), [1] * len(terms),
             color=term_colors, alpha=0.75, height=0.5,
             edgecolor="white", linewidth=0.8, zorder=3)
    for i, t in enumerate(terms):
        label = f'"{t}"  —  {"covered ✓" if t in covered else "missing ✗"}'
        ax2.text(0.05, i, label, va="center", fontsize=9,
                 color=TEXT if t in covered else MUTED)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title(f"Query Term Coverage  ({len(covered)}/{len(terms)} terms found)",
                  fontsize=10, fontweight="bold", color=TEXT, pad=4)
    _style(ax2)
    ax2.grid(False)

    # ── Sub-plot C: Specificity on/off-topic comparison ───────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    cat = ["On-topic\nsimilarity", "Off-topic\n(control queries)", "Specificity\nscore"]
    vals = [on_t, off_t, spec]
    bar_colors = [BLUE, RED, TEAL]
    b3 = ax3.bar(np.arange(3), vals, color=bar_colors, width=0.5, zorder=3,
                 edgecolor="white", linewidth=0.8)
    for bar, v in zip(b3, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                 color=TEXT, fontweight="bold")
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(cat, fontsize=8.5, color=TEXT)
    ax3.set_ylim(0, 1.25)
    ax3.set_ylabel("Cosine Similarity", fontsize=9, color=TEXT)
    ax3.set_title("Specificity Breakdown\n(on-topic vs off-topic similarity)",
                  fontsize=10, fontweight="bold", color=TEXT, pad=4)
    ax3.text(0.5, -0.16,
             "Specificity = max(0, (on-topic − off-topic) / on-topic)",
             transform=ax3.transAxes, ha="center", fontsize=7.5,
             color=MUTED, style="italic")
    _style(ax3)

    fig.suptitle("Query Relevance Evaluation — Sub-Dimension Breakdown",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.02)
    fig.text(0.5, 0.98,
             f"Final score: {total:.4f}  |  "
             "Semantic relevance 35% + sub-question coverage 40% + specificity 25%",
             ha="center", fontsize=9, color=SUBTITLE, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "chart_sci2_query_relevance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] {out}")
    return out


# =============================================================================
# Chart 3 — Perspective Diversity
# =============================================================================
def chart_perspective_diversity(data: dict, out_dir: Path) -> Path:
    counts = np.array(data["article_counts"])
    probs  = data["probabilities"]
    K      = data["cluster_count"]
    H      = data["shannon_entropy_bits"]
    H_max  = data["max_entropy_bits"]
    H_norm = data["normalised_entropy"]

    cluster_labels = [
        "Tech optimists\n(AI augments developers)",
        "Labour economists\n(displacement risk)",
        "Policy advocates\n(reskilling needed)",
        "Emerging view\n(AI & hiring inequality)",
    ][:K]
    while len(cluster_labels) < K:
        cluster_labels.append(f"Cluster {len(cluster_labels)+1}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(BG)

    # ── Left: cluster article bars ────────────────────────────────────────────
    ax.set_facecolor(BG)
    cluster_colors = [BLUE, MID_BLUE, ORANGE, LIGHT][:K]
    x = np.arange(K)
    bars = ax.bar(x, counts, color=cluster_colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8)
    for bar, c, p in zip(bars, counts, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.12,
                f"{int(c)} articles\n({p:.1%} of corpus)",
                ha="center", va="bottom", fontsize=9, color=TEXT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, fontsize=9, color=TEXT)
    ax.set_ylabel("Number of Articles", fontsize=10, color=TEXT)
    ax.set_ylim(0, max(counts) * 1.7)
    ax.set_title("Article Distribution Across NLI-Identified Viewpoints",
                 fontsize=11, fontweight="bold", color=TEXT, pad=4)
    _style(ax)

    # ── Right: entropy visualisation ─────────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.axis("off")

    # Draw a gauge: normalised entropy as fraction of a rectangle
    gauge_h = 0.18
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.58), 0.90, gauge_h,
        boxstyle="round,pad=0.01", facecolor=GRID, edgecolor=GRID))
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.58), 0.90 * H_norm, gauge_h,
        boxstyle="round,pad=0.01", facecolor=TEAL, edgecolor=TEAL, alpha=0.85))
    ax2.text(0.50, 0.58 + gauge_h / 2,
             f"H_norm = {H_norm:.4f}",
             ha="center", va="center", fontsize=13,
             color="white", fontweight="bold")
    ax2.text(0.05, 0.58 - 0.05, "0.0 (one cluster\ndominates)",
             ha="left", fontsize=8, color=MUTED)
    ax2.text(0.95, 0.58 - 0.05, "1.0 (perfectly\nbalanced)",
             ha="right", fontsize=8, color=MUTED)

    # Entropy formula + values
    formula_lines = [
        ("Shannon Entropy (Shannon, 1948):", 0.50, 12, True),
        ("H = −Σ pᵢ · log₂(pᵢ)", 0.43, 11, False),
        (f"H = {H:.4f} bits", 0.36, 10, False),
        (f"H_max = log₂({K}) = {H_max:.4f} bits", 0.29, 10, False),
        (f"H_norm = {H:.4f} / {H_max:.4f} = {H_norm:.4f}", 0.21, 10, False),
        ("Interpretation:", 0.12, 9.5, True),
        (f"Near-uniform distribution across {K} viewpoints.", 0.06, 9, False),
        ("No single cluster dominates the discourse.", 0.00, 9, False),
    ]
    for line, y_pos, fs, bold in formula_lines:
        ax2.text(0.5, y_pos + 0.10, line, ha="center", va="bottom",
                 fontsize=fs, color=TEXT,
                 fontweight="bold" if bold else "normal")

    _titles(fig, ax,
            "Perspective Diversity — Shannon Entropy Breakdown",
            f"H_norm = {H_norm:.4f}  (1.0 = perfectly balanced across {K} viewpoints)  |  "
            "Evo (single narrative) scores H_norm = 0.000 by design")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "chart_sci3_perspective_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] {out}")
    return out


# =============================================================================
# Chart 4 — Sentiment Trend Coherence
# =============================================================================
def chart_sentiment_trend(data: dict, out_dir: Path) -> Path:
    t = np.array(data["time_indices"])
    s = np.array(data["sentiment_values"])
    r = data["pearson_r"]
    n = data["n_buckets"]

    bucket_labels = [f"Bucket {int(ti)}" for ti in t]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Regression line
    m, b = np.polyfit(t, s, 1)
    x_line = np.linspace(t.min() - 0.2, t.max() + 0.2, 200)
    ax.plot(x_line, m * x_line + b, color=ORANGE, linewidth=2.2,
            linestyle="-", zorder=4, alpha=0.85, label=f"Linear trend (slope = {m:+.3f})")

    # Zero line
    ax.axhline(0, color=GRID, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(t.max() + 0.22, 0.005, "neutral sentiment (0.0)",
            fontsize=8.5, color=MUTED, va="bottom")

    # Scatter + value labels
    ax.scatter(t, s, color=BLUE, s=120, zorder=6,
               edgecolors="white", linewidths=1.2, label="Observed mean sentiment")
    for xi, si in zip(t, s):
        valign = "bottom" if si >= 0 else "top"
        offset = 0.012 if si >= 0 else -0.012
        ax.text(xi, si + offset, f"{si:+.2f}",
                ha="center", va=valign, fontsize=10.5,
                color=TEXT, fontweight="bold")

    # Pearson r annotation box
    strength = "strong" if abs(r) >= 0.70 else "moderate" if abs(r) >= 0.40 else "weak"
    direction = "positive" if r > 0 else "negative"
    ax.text(0.03, 0.97,
            f"Pearson r = {r:.4f}\n"
            f"{strength.capitalize()} {direction} linear trend\n"
            f"(Cohen 1988: strong ≥ 0.70)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, alpha=0.95))

    period_map = {0: "Days 1–67\n(early period)",
                  1: "Days 68–134\n(mid period)",
                  2: "Days 135–200\n(late period)"}
    combined_labels = [f"Bucket {int(ti)}\n{period_map.get(int(ti), '')}" for ti in t]
    ax.set_xticks(t)
    ax.set_xticklabels(combined_labels, fontsize=9.5, color=TEXT, linespacing=1.4)
    ax.set_ylabel("Mean Sentiment Score\n(−1 = very negative,  +1 = very positive)",
                  fontsize=10, color=TEXT)
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax.legend(fontsize=9, framealpha=0.9, facecolor=BG, edgecolor=GRID,
              loc="lower right", labelcolor=TEXT)

    _titles(fig, ax,
            "Sentiment Trend Coherence — Pearson Correlation",
            f"r = {r:.4f}  ({strength} {direction} trend)  |  "
            "Sentiment shifted from net-negative to net-positive across the 200-day window")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "chart_sci4_sentiment_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] {out}")
    return out


# =============================================================================
# Chart 5 — Event Correlation (Precision@K)
# =============================================================================
def chart_event_correlation(data: dict, out_dir: Path) -> Path:
    K        = data["K"]
    matched  = data["matched_events"]
    pk       = data["precision_at_k"]
    mean_p   = data["mean_correlated_event_plausibility"]

    matched_ids = {m["bucket_id"] for m in matched}

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    TL_Y = 0.50   # y-coordinate of the timeline

    # Horizontal timeline
    ax.axhline(TL_Y, xmin=0.04, xmax=0.96, color="#CCCCCC", linewidth=2.5, zorder=2)

    for bid in range(K):
        x_pos = bid

        # Bucket header — always above the timeline dot
        ax.text(x_pos, TL_Y + 0.13,
                f"Inflection {bid + 1}\n(Bucket {bid})",
                ha="center", va="bottom", fontsize=9.5,
                color=TEXT, fontweight="bold")

        if bid in matched_ids:
            ev    = next(m for m in matched if m["bucket_id"] == bid)
            plaus = ev["plausibility"]
            desc  = ev["description"]

            ax.scatter(x_pos, TL_Y, s=300, color=TEAL,
                       edgecolors="white", linewidths=1.8, zorder=6)

            # All matched callouts go BELOW the timeline
            wrapped = "\n".join(textwrap.wrap(desc, width=32))
            label   = wrapped + f"\n\nPlausibility: {plaus:.2f}"

            y_box = TL_Y - 0.52
            ax.annotate("",
                xy=(x_pos, TL_Y - 0.09),
                xytext=(x_pos, y_box + 0.01),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))

            ax.text(x_pos, y_box, label,
                    ha="center", va="top", fontsize=9, color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.50", facecolor=TEAL,
                              edgecolor="#1A9E94", alpha=0.18, linewidth=1.2))

        else:
            # Unmatched — hollow circle; label goes BELOW too (shorter)
            ax.scatter(x_pos, TL_Y, s=220, facecolors="none",
                       edgecolors=MUTED, linewidths=2.2, zorder=6)
            ax.text(x_pos, TL_Y - 0.18,
                    "Gradual discourse shift\n(no single triggering event)",
                    ha="center", va="top", fontsize=9,
                    color=MUTED, style="italic")

    # P@K summary box — top right, well above all callouts
    mean_p_str = f"{mean_p:.2f}" if mean_p is not None else "N/A"
    ax.text(0.98, 0.97,
            f"P@{K} = {data['matched']}/{K} = {pk:.4f}\n"
            f"Mean plausibility = {mean_p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, alpha=0.95))

    # Legend — top left, away from summary box
    handles = [
        mpatches.Patch(facecolor=TEAL, alpha=0.85, label="Matched — GDELT event found"),
        mpatches.Patch(facecolor="none", edgecolor=MUTED, linewidth=2.2,
                       label="Unmatched — gradual discourse shift"),
    ]
    ax.legend(handles=handles, fontsize=9.5, framealpha=0.9,
              facecolor=BG, edgecolor=GRID,
              loc="upper left", labelcolor=TEXT)

    ax.set_xlim(-0.6, K - 0.4)
    ax.set_ylim(-0.85, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    _titles(fig, ax,
            "Event Correlation — Precision@K",
            f"P@{K} = {pk:.4f}  |  "
            f"{data['matched']} of {K} detected inflection points linked to a real-world GDELT event")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "chart_sci5_event_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] {out}")
    return out


# =============================================================================
# Main
# =============================================================================
def generate_all(sci_path: Path):
    with open(sci_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"\n[INFO] Claim: {results['meta']['claim']}")
    print(f"[INFO] Generating 5 individual charts → {CHARTS_DIR}\n")

    chart_groundedness(results["groundedness"], CHARTS_DIR)
    chart_query_relevance(results["query_relevance"], CHARTS_DIR)
    chart_perspective_diversity(results["perspective_diversity"], CHARTS_DIR)
    chart_sentiment_trend(results["sentiment_trend"], CHARTS_DIR)
    chart_event_correlation(results["event_correlation"], CHARTS_DIR)

    print("\n[DONE] All 5 charts saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Path to revamped_scientific_*.json. Auto-discovers latest if omitted.")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sci_path = Path(args.input) if args.input else _find_latest_scientific()
    if not sci_path or not sci_path.exists():
        print("[ERROR] No scientific evaluation results found. "
              "Run evaluate_revamped_scientific.py first.")
        sys.exit(1)

    print(f"[INFO] Loading: {sci_path.name}")
    generate_all(sci_path)


if __name__ == "__main__":
    main()
