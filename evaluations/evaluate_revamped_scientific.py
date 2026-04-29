"""
evaluate_revamped_scientific.py
================================
Runs scientifically grounded evaluation of Evo-Revamped using the same
standard evaluators as Evo (GroundednessEvaluator, RelevanceEvaluator),
supplemented with standard statistical measures for Revamped-exclusive
analytical capabilities:

  Metric 1  — Groundedness Score          (hybrid NLI + SBERT, same as Evo)
  Metric 2  — Query Relevance Score       (semantic + coverage + specificity, same as Evo)
  Metric 3  — Perspective Diversity       (Shannon Entropy, normalised)
  Metric 4  — Sentiment Trend Coherence   (Pearson correlation r)
  Metric 5  — Event Correlation           (Precision@K)

All five metrics are established in the NLP/IR literature.
No weights are invented: Metrics 1-2 use the same methodology as Evo;
Metrics 3-5 use named, citable standard measures.

Usage:
    python evaluations/evaluate_revamped_scientific.py
    python evaluations/evaluate_revamped_scientific.py --input path/to/revamped_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project roots on path ─────────────────────────────────────────────────────
_REVAMPED_ROOT = Path(__file__).resolve().parent.parent
_EVO_BACKEND   = _REVAMPED_ROOT.parent / "Evo" / "backend"

for p in [str(_REVAMPED_ROOT), str(_EVO_BACKEND)]:
    if p not in sys.path:
        sys.path.insert(0, p)

_EVAL_DIR   = Path(__file__).parent
RESULTS_DIR = _EVAL_DIR / "evaluation_results"
CHARTS_DIR  = _EVAL_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ── Load Evo's evaluators (same code, no modification) ────────────────────────
from groundedness_evaluator import GroundednessEvaluator   # noqa: E402
from relevance_evaluator    import RelevanceEvaluator       # noqa: E402


# =============================================================================
# Metric 3 — Perspective Diversity via Shannon Entropy
# =============================================================================
# Source: Shannon, C. E. (1948). "A mathematical theory of communication."
#         Bell System Technical Journal, 27(3), 379–423.
#
# Article counts per cluster form a discrete probability distribution.
# H = -Σ p_i * log2(p_i)   (bits of entropy)
# H_norm = H / log2(K)     (normalised to [0, 1] relative to maximum possible
#                            entropy for K equally-sized clusters)
#
# A uniform distribution (all clusters equal size) achieves H_norm = 1.0.
# A degenerate distribution (one cluster dominates) approaches H_norm = 0.0.
# =============================================================================

def compute_shannon_entropy(clusters: list[dict]) -> dict:
    counts = np.array([max(c.get("article_count", 1), 1) for c in clusters], dtype=float)
    total  = counts.sum()
    probs  = counts / total

    # Shannon entropy in bits
    H = -float(np.sum(probs * np.log2(probs + 1e-12)))

    # Maximum possible entropy for K clusters (uniform distribution)
    K     = len(clusters)
    H_max = math.log2(K) if K > 1 else 1.0

    # Normalised entropy
    H_norm = round(H / H_max, 4)

    return {
        "metric": "Perspective Diversity (Shannon Entropy)",
        "cluster_count": K,
        "article_counts": counts.tolist(),
        "probabilities": [round(float(p), 4) for p in probs],
        "shannon_entropy_bits": round(H, 4),
        "max_entropy_bits": round(H_max, 4),
        "normalised_entropy": H_norm,
        "interpretation": (
            f"H = {H:.4f} bits out of a maximum {H_max:.4f} bits for {K} clusters. "
            f"Normalised H = {H_norm:.4f} — a value of 1.0 would indicate perfectly "
            "equal article distribution across all clusters (maximum diversity); "
            "a value approaching 0.0 would indicate a single dominant cluster."
        ),
    }


# =============================================================================
# Metric 4 — Sentiment Trend Coherence via Pearson Correlation
# =============================================================================
# Source: Pearson, K. (1895). "Note on regression and inheritance in the case
#         of two parents." Proceedings of the Royal Society of London, 58, 240-242.
#
# Tests whether the mean sentiment across time buckets follows a statistically
# coherent linear trend. r is the Pearson product-moment correlation coefficient
# between time indices [0, 1, 2, ...] and mean_sentiment values.
#
# r ∈ [−1, 1]:
#   r ≈ +1 → strong positive trend (consistently improving sentiment)
#   r ≈ −1 → strong negative trend (consistently worsening sentiment)
#   r ≈  0 → no linear trend (sentiment is unstable or flat)
# =============================================================================

def compute_sentiment_pearson(sentiment_timeline: list[dict]) -> dict:
    if len(sentiment_timeline) < 2:
        return {
            "metric": "Sentiment Trend Coherence (Pearson r)",
            "n_buckets": len(sentiment_timeline),
            "pearson_r": None,
            "interpretation": "Insufficient data points for correlation (need ≥ 2 buckets).",
        }

    t = np.array([b["bucket_id"] for b in sentiment_timeline], dtype=float)
    s = np.array([b["mean_sentiment"] for b in sentiment_timeline], dtype=float)

    # Pearson r from numpy
    r = float(np.corrcoef(t, s)[0, 1])

    # Magnitude interpretation (Cohen 1988 conventions for r)
    abs_r = abs(r)
    if abs_r >= 0.70:
        strength = "strong"
    elif abs_r >= 0.40:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if r > 0 else "negative"

    return {
        "metric": "Sentiment Trend Coherence (Pearson r)",
        "n_buckets": len(t),
        "time_indices": t.tolist(),
        "sentiment_values": s.tolist(),
        "pearson_r": round(r, 4),
        "interpretation": (
            f"r = {r:.4f} — {strength} {direction} linear relationship between time and sentiment. "
            + (
                "Sentiment improved consistently across the observation window, "
                "suggesting a coherent narrative shift rather than random fluctuation."
                if r > 0.40
                else (
                    "Sentiment declined consistently across the observation window."
                    if r < -0.40
                    else "No strong linear sentiment trend detected across time buckets."
                )
            )
        ),
    }


# =============================================================================
# Metric 5 — Event Correlation via Precision@K
# =============================================================================
# Source: Manning, C. D., Raghavan, P., & Schütze, H. (2008).
#         Introduction to Information Retrieval (p. 158). Cambridge University Press.
#
# Among the K detected inflection points, what fraction were successfully
# linked to a verifiable real-world event (non-null correlated_event)?
#
# P@K = |{relevant inflections in top K}| / K
#
# This is the standard IR metric for measuring retrieval precision at a fixed
# cutoff. Here K = total number of detected inflection points (all are evaluated).
# =============================================================================

def compute_precision_at_k(inflection_cards: list[dict]) -> dict:
    K = len(inflection_cards)
    if K == 0:
        return {
            "metric": "Event Correlation Precision (P@K)",
            "K": 0,
            "matched": 0,
            "precision_at_k": None,
            "interpretation": "No inflection points detected — P@K undefined.",
        }

    matched = [ip for ip in inflection_cards if ip.get("correlated_event") is not None]
    M       = len(matched)
    pk      = round(M / K, 4)

    plausibilities = [
        ip["correlated_event"]["plausibility_score"]
        for ip in matched
        if isinstance(ip.get("correlated_event"), dict)
           and ip["correlated_event"].get("plausibility_score") is not None
    ]
    mean_plausibility = round(float(np.mean(plausibilities)), 4) if plausibilities else None

    return {
        "metric": "Event Correlation Precision (P@K)",
        "K": K,
        "matched": M,
        "unmatched": K - M,
        "precision_at_k": pk,
        "mean_correlated_event_plausibility": mean_plausibility,
        "matched_events": [
            {
                "bucket_id":   ip["bucket_id"],
                "description": ip["correlated_event"].get("description", "")[:120],
                "plausibility": ip["correlated_event"].get("plausibility_score"),
            }
            for ip in matched
        ],
        "interpretation": (
            f"P@{K} = {M}/{K} = {pk:.4f}. "
            f"{M} of {K} detected inflection points were successfully linked to "
            "a verifiable real-world event via GDELT query. "
            + (
                f"Mean LLM-assessed plausibility of matched events: {mean_plausibility:.2f}."
                if mean_plausibility is not None else ""
            )
        ),
    }


# =============================================================================
# Main evaluation runner
# =============================================================================

def _find_latest_dryrun() -> Path | None:
    rev_dir = _EVAL_DIR / "revamped_outputs"
    if not rev_dir.exists():
        return None
    # prefer dry-run files
    files = sorted(rev_dir.glob("*_dryrun_*.json"))
    return files[-1] if files else None


def run_scientific_evaluation(revamped_path: Path) -> dict:
    with open(revamped_path, encoding="utf-8") as f:
        state = json.load(f)

    report   = state.get("narrative_report", state)
    clusters = (
        report.get("perspective_landscape")
        or state.get("perspective_clusters")
        or []
    )
    sentiment_timeline  = report.get("sentiment_timeline", [])
    inflection_cards    = report.get("inflection_point_cards", [])
    claim               = state.get("claim", report.get("claim_snapshot", {}).get("claim", ""))
    summary             = report.get("narrative_intelligence_summary", "")

    # ── Evidence for groundedness: key claims extracted from all clusters ─────
    # These are the factual assertions the LLM had access to when generating
    # the narrative summary — the same role that SERP snippets play in Evo.
    evidence_snippets = [
        claim_text
        for c in clusters
        for claim_text in c.get("key_claims", [])
    ]
    # Also include cluster labels as lightweight evidence
    evidence_snippets += [c.get("label", "") for c in clusters if c.get("label")]

    # ── Metric 1: Groundedness (same evaluator as Evo) ─────────────────────────
    print("\n" + "="*60)
    print("METRIC 1 — Groundedness (Hybrid NLI + SBERT)")
    print("="*60)
    g_eval  = GroundednessEvaluator(threshold=0.30, nli_weight=0.4, sim_weight=0.6)
    g_result = g_eval.evaluate(summary, evidence_snippets)

    # ── Metric 2: Query Relevance (same evaluator as Evo, local fallback) ─────
    print("\n" + "="*60)
    print("METRIC 2 — Query Relevance (Semantic + Coverage + Specificity)")
    print("="*60)
    r_eval  = RelevanceEvaluator()
    # Build output sections from the report's main text components
    perspective_text = " ".join(
        f"{c.get('label','')}: {' '.join(c.get('key_claims', []))}"
        for c in clusters
    )
    output_sections = {
        "narrative_summary": summary,
        "perspective_clusters": perspective_text,
    }
    r_result = r_eval.evaluate(
        query=claim,
        output_sections=output_sections,
        groq_client=None,   # no API keys in dry-run; uses local fallback
    )

    # ── Metric 3: Perspective Diversity (Shannon Entropy) ─────────────────────
    print("\n" + "="*60)
    print("METRIC 3 — Perspective Diversity (Shannon Entropy)")
    print("="*60)
    e_result = compute_shannon_entropy(clusters)
    print(f"  Shannon H       = {e_result['shannon_entropy_bits']:.4f} bits")
    print(f"  Max possible H  = {e_result['max_entropy_bits']:.4f} bits")
    print(f"  Normalised H    = {e_result['normalised_entropy']:.4f}")

    # ── Metric 4: Sentiment Trend Coherence (Pearson r) ───────────────────────
    print("\n" + "="*60)
    print("METRIC 4 — Sentiment Trend Coherence (Pearson r)")
    print("="*60)
    s_result = compute_sentiment_pearson(sentiment_timeline)
    print(f"  Pearson r = {s_result['pearson_r']}")
    print(f"  {s_result['interpretation']}")

    # ── Metric 5: Event Correlation (Precision@K) ─────────────────────────────
    print("\n" + "="*60)
    print("METRIC 5 — Event Correlation (Precision@K)")
    print("="*60)
    p_result = compute_precision_at_k(inflection_cards)
    print(f"  P@{p_result['K']} = {p_result['precision_at_k']}")
    print(f"  {p_result['interpretation']}")

    # ── Assemble results ───────────────────────────────────────────────────────
    results = {
        "meta": {
            "claim":         claim,
            "input_file":    str(revamped_path),
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "evaluation_note": (
                "Metrics 1-2 use identical evaluators to Evo's evaluation pipeline. "
                "Metrics 3-5 use standard NLP/IR measures: Shannon Entropy (Shannon 1948), "
                "Pearson correlation (Pearson 1895), and Precision@K (Manning et al. 2008)."
            ),
        },
        "groundedness":             g_result,
        "query_relevance":          r_result,
        "perspective_diversity":    e_result,
        "sentiment_trend":          s_result,
        "event_correlation":        p_result,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"revamped_scientific_{ts}.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[SAVED] {out_path}")
    return results


def _print_summary(results: dict):
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    g  = results["groundedness"]
    r  = results["query_relevance"]
    e  = results["perspective_diversity"]
    s  = results["sentiment_trend"]
    pk = results["event_correlation"]

    print("\n" + "="*64)
    print("  EVO-REVAMPED — SCIENTIFIC EVALUATION SUMMARY")
    print("="*64)
    print(f"  Claim: {results['meta']['claim']}")
    print()
    print(f"  Groundedness Score          : {g['groundedness_score']:.4f}  "
          f"({g['grounded_claims']}/{g['total_claims']} claims grounded)")
    print(f"  Query Relevance Score       : {r['query_relevance_score']:.4f}  "
          f"(semantic={r['semantic_relevance']['overall']:.4f}, "
          f"coverage={r['coverage']['coverage_score']:.4f}, "
          f"specificity={r['specificity']['specificity_score']:.4f})")
    print(f"  Perspective Diversity (H)   : {e['normalised_entropy']:.4f}  "
          f"(Shannon H={e['shannon_entropy_bits']:.4f} bits, K={e['cluster_count']} clusters)")
    print(f"  Sentiment Trend (Pearson r) : {s['pearson_r']}  ({s['interpretation'][:60]}...)")
    print(f"  Event Correlation (P@K)     : {pk['precision_at_k']}  "
          f"({pk['matched']}/{pk['K']} inflection points matched)")
    print("="*64)


def main():
    parser = argparse.ArgumentParser(
        description="Scientific evaluation of Evo-Revamped using standard NLP/IR metrics."
    )
    parser.add_argument("--input", default=None,
                        help="Path to revamped_*_dryrun_*.json. Auto-discovers latest if omitted.")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rev_path = Path(args.input) if args.input else _find_latest_dryrun()
    if not rev_path or not rev_path.exists():
        print("[ERROR] No dry-run Evo-Revamped output found. "
              "Run: python evaluations/generate_revamped_output.py --dry-run")
        sys.exit(1)

    print(f"[INFO] Evaluating: {rev_path.name}")
    results = run_scientific_evaluation(rev_path)
    _print_summary(results)


if __name__ == "__main__":
    main()
