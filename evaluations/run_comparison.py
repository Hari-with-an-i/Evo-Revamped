"""
run_comparison.py — Orchestrator for the full Evo vs Evo-Revamped comparison
=============================================================================
Ties together:
  1. The Evo legacy groundedness JSON (already in Evo/backend/evaluation_results/)
  2. An Evo-Revamped output JSON (produced by generate_revamped_output.py)
  3. The comparative_evaluator metrics engine
  4. Chart generation

Usage:
    # Full automated run (dry-run Revamped + auto-discover Evo results):
    python evaluations/run_comparison.py --auto

    # With explicit paths:
    python evaluations/run_comparison.py \\
        --evo-results "../Evo/backend/evaluation_results/groundedness_trend_20260302_011242.json" \\
        --revamped-report "evaluations/revamped_outputs/revamped_ai_will_replace_*.json" \\
        --claim "AI will replace all junior software developers by 2030"

    # Generate dry-run Revamped output then compare:
    python evaluations/run_comparison.py --auto --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import glob

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Relative sibling imports
_EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(_EVAL_DIR))

from comparative_evaluator import run_full_comparison

COMPARISON_DIR = _EVAL_DIR / "comparison_results"
COMPARISON_DIR.mkdir(exist_ok=True)

# ── Default Evo result path (already exists) ──────────────────────────────────
_DEFAULT_EVO_RESULTS = (
    _ROOT.parent / "Evo" / "backend" / "evaluation_results"
    / "groundedness_trend_20260302_011242.json"
)


def _resolve_glob(pattern: str) -> Path | None:
    """Resolve a glob pattern to the most recent matching file."""
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _find_latest_revamped() -> Path | None:
    rev_dir = _EVAL_DIR / "revamped_outputs"
    if not rev_dir.exists():
        return None
    files = sorted(rev_dir.glob("revamped_*.json"))
    return files[-1] if files else None


def _find_latest_evo() -> Path | None:
    evo_dir = _ROOT.parent / "Evo" / "backend" / "evaluation_results"
    if evo_dir.exists():
        files = sorted(evo_dir.glob("groundedness_trend_*.json"))
        if files:
            return files[-1]
    if _DEFAULT_EVO_RESULTS.exists():
        return _DEFAULT_EVO_RESULTS
    return None


def run_auto(dry_run: bool = False):
    """
    Full automated pipeline:
      1. Generate Revamped output (dry-run or real)
      2. Auto-discover Evo results
      3. Run comparison
      4. Generate charts
    """
    print("=" * 60)
    print("STEP 1 — Generating Evo-Revamped output")
    print("=" * 60)

    import generate_revamped_output as gro  # sibling module

    if dry_run:
        print("[INFO] Dry-run mode — generating mock output (no API calls)")
        claim = gro.TEST_CLAIMS[0]
        state = gro.make_mock_output(claim)
        rev_path = gro.save_output(state, claim, dry_run=True)
    else:
        claim = gro.TEST_CLAIMS[0]
        state = gro.run_pipeline(claim)
        rev_path = gro.save_output(state, claim)

    print("\n" + "=" * 60)
    print("STEP 2 — Locating Evo legacy evaluation results")
    print("=" * 60)

    evo_path = _find_latest_evo()
    if not evo_path:
        print("[ERROR] Could not find Evo groundedness JSON. "
              "Ensure d:\\Projects\\Evo\\backend\\evaluation_results\\ exists.")
        sys.exit(1)
    print(f"[OK] Using Evo results: {evo_path}")

    print("\n" + "=" * 60)
    print("STEP 3 — Running comparative evaluation")
    print("=" * 60)

    result = run_full_comparison(
        evo_groundedness_path=str(evo_path),
        revamped_report_path=str(rev_path),
        claim=claim,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = COMPARISON_DIR / f"comparison_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVED] Comparison report: {out_path}")

    _print_summary(result)

    print("\n" + "=" * 60)
    print("STEP 4 — Generating charts")
    print("=" * 60)

    import generate_comparison_charts as gcc  # sibling module

    charts_dir = _EVAL_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)

    gcc.chart_per_metric_bars(result, charts_dir)
    gcc.chart_radar(result, charts_dir)
    gcc.chart_composite(result, charts_dir)
    gcc.chart_sentiment_timeline(state, charts_dir)
    gcc.chart_perspective_clusters(state, charts_dir)

    print(f"\n[DONE] All charts saved to {charts_dir}/")
    return result


def run_explicit(evo_path: str, rev_path: str, claim: str):
    """Run comparison with explicit file paths."""
    result = run_full_comparison(
        evo_groundedness_path=evo_path,
        revamped_report_path=rev_path,
        claim=claim,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = COMPARISON_DIR / f"comparison_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVED] Comparison report: {out_path}")
    _print_summary(result)

    # Charts
    import generate_comparison_charts as gcc
    charts_dir = _EVAL_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)
    gcc.chart_per_metric_bars(result, charts_dir)
    gcc.chart_radar(result, charts_dir)
    gcc.chart_composite(result, charts_dir)

    with open(rev_path, encoding="utf-8") as f:
        rev_data = json.load(f)
    gcc.chart_sentiment_timeline(rev_data, charts_dir)
    gcc.chart_perspective_clusters(rev_data, charts_dir)

    return result


def _print_summary(result: dict):
    cs = result["composite_scores"]
    pm = result["per_metric_scores"]
    print("\n" + "─" * 56)
    print(f"{'METRIC':<30} {'EVO':>8} {'REVAMPED':>10}")
    print("─" * 56)
    labels = {
        "groundedness": "Groundedness",
        "temporal_coverage": "Temporal Coverage",
        "perspective_diversity": "Perspective Diversity",
        "event_correlation": "Event Correlation",
    }
    for k, label in labels.items():
        print(f"{label:<30} {pm['evo'][k]:>8.4f} {pm['revamped'][k]:>10.4f}")
    print("─" * 56)
    print(f"{'COMPOSITE SCORE':<30} {cs['evo']:>8.4f} {cs['revamped']:>10.4f}")
    print("─" * 56)
    print(f"Improvement: +{cs['improvement_pct']:.1f}%")
    print()
    print(result["verdict"])
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate full Evo vs Evo-Revamped comparison."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--auto",  action="store_true",
                     help="Fully automated run (generate Revamped output + compare + chart)")
    grp.add_argument("--evo-results",  metavar="PATH",
                     help="Explicit path to Evo groundedness JSON")

    parser.add_argument("--revamped-report", metavar="PATH",
                        help="Explicit path to Revamped output JSON")
    parser.add_argument("--claim",  default="",
                        help="Claim text for metadata")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock Revamped output instead of calling APIs")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.auto:
        run_auto(dry_run=args.dry_run)
    else:
        if not args.revamped_report:
            parser.error("--revamped-report is required with --evo-results")
        rev_path = _resolve_glob(args.revamped_report)
        if not rev_path:
            parser.error(f"No file found matching: {args.revamped_report}")
        run_explicit(args.evo_results, str(rev_path), args.claim)


if __name__ == "__main__":
    main()
