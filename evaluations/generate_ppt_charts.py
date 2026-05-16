"""
Generate 3 clean evaluation charts from scientific evaluation results.

Charts:
  ppt_chart1_groundedness.png   — grounded vs ungrounded claims per topic category
  ppt_chart2_perspectives.png   — distinct perspectives found per topic category
  ppt_chart3_relevance.png      — query relevance score per topic category
"""

import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts", "ppt")

CATEGORY_MAP = {
    "prices to rise over 30%":  "Geopolitics",
    "surged over 30%":          "Trade & Economics",
    "AI will replace":          "AI & Automation",
    "Cryptocurrency regulation": "Crypto & Finance",
}


def assign_category(claim: str) -> str:
    for substring, label in CATEGORY_MAP.items():
        if substring.lower() in claim.lower():
            return label
    return "Other"


def load_all_results(results_dir: str) -> list[dict]:
    pattern = os.path.join(results_dir, "revamped_scientific_*.json")
    files = sorted(glob.glob(pattern))
    latest: dict[str, dict] = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        latest[data["meta"]["claim"]] = data
    return list(latest.values())


# ---------------------------------------------------------------------------
# Chart 1 — Groundedness
# ---------------------------------------------------------------------------

def chart_groundedness(results: list[dict], out_dir: str) -> str:
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = assign_category(r["meta"]["claim"])
        g = r["groundedness"]
        if cat not in by_cat:
            by_cat[cat] = {"grounded": 0, "total": 0}
        by_cat[cat]["grounded"] += g["grounded_claims"]
        by_cat[cat]["total"]    += g["total_claims"]

    categories = list(by_cat.keys())
    grounded   = [by_cat[c]["grounded"] for c in categories]
    ungrounded = [by_cat[c]["total"] - by_cat[c]["grounded"] for c in categories]

    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_g = ax.bar(x, grounded,   width, label="Grounded",     color="C0")
    bars_u = ax.bar(x, ungrounded, width, label="Not grounded", color="C3",
                    bottom=grounded)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Number of Claims")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    path = os.path.join(out_dir, "ppt_chart1_groundedness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 2 — Perspective Diversity
# ---------------------------------------------------------------------------

def chart_perspective_diversity(results: list[dict], out_dir: str) -> str:
    by_cat: dict[str, int] = {}
    for r in results:
        cat = assign_category(r["meta"]["claim"])
        by_cat[cat] = r["perspective_diversity"]["cluster_count"]

    categories = list(by_cat.keys())
    counts = [by_cat[c] for c in categories]

    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x, counts, width, color="C0")

    ax.axhline(3, color="grey", linestyle="--", linewidth=1, label="Min. target (3)")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Perspectives Found")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0, max(counts) + 1.5)
    ax.legend(frameon=False, fontsize=9)

    for i, count in enumerate(counts):
        ax.text(x[i], count + 0.1, str(count), ha="center", va="bottom",
                fontsize=9, color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    path = os.path.join(out_dir, "ppt_chart2_perspectives.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 3 — Query Relevance
# ---------------------------------------------------------------------------

def chart_query_relevance(results: list[dict], out_dir: str) -> str:
    by_cat: dict[str, float] = {}
    for r in results:
        cat = assign_category(r["meta"]["claim"])
        by_cat[cat] = r["query_relevance"]["query_relevance_score"]

    categories = list(by_cat.keys())
    scores = [by_cat[c] for c in categories]

    def bar_color(s: float) -> str:
        if s >= 0.80:
            return "C0"
        if s >= 0.60:
            return "C1"
        return "C3"

    colors = [bar_color(s) for s in scores]
    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x, scores, width, color=colors)

    ax.axhline(0.80, color="grey", linestyle="--", linewidth=1)
    ax.text(len(categories) - 0.5, 0.81, "0.80", fontsize=8, color="grey",
            va="bottom", ha="right")

    for i, score in enumerate(scores):
        ax.text(x[i], score + 0.01, f"{score:.2f}", ha="center", va="bottom",
                fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Relevance Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    path = os.path.join(out_dir, "ppt_chart3_relevance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} unique queries")

    os.makedirs(OUT_DIR, exist_ok=True)

    p1 = chart_groundedness(results, OUT_DIR)
    print(f"Saved: {p1}")

    p2 = chart_perspective_diversity(results, OUT_DIR)
    print(f"Saved: {p2}")

    p3 = chart_query_relevance(results, OUT_DIR)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
