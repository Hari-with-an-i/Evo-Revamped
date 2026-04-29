"""
direct_run.py — Run pipeline with forced exit after first analyst completion.
Bypasses the orchestrator's looping behaviour by catching the state the moment
analysis_complete=True is set, then immediately saves and exits.
"""
import json
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from src.config import config
from src.graphs import build_graph

CLAIM = "Climate change is causing an irreversible economic crisis that current financial models cannot predict"

OUTPUT_DIR = Path(__file__).parent / "revamped_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def _slug(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", claim.lower())[:60].strip("_")

def run():
    graph = build_graph()
    initial_state = {
        "messages": [HumanMessage(content=CLAIM)],
        "raw_input": CLAIM,
        "input_type": "claim",
        "claim": "",
        "entities": [],
        "timeframe": None,
        "claim_complexity": None,
        "context_summary": None,
        "retrieved_articles": [],
        "anchor_articles": [],
        "targeted_queries": [],
        "retrieval_loop_count": 0,
        "retrieval_complete": False,
        "perspective_clusters": [],
        "coverage_gaps": [],
        "evaluation_ready": None,
        "primary_evidence_exists": None,
        "primary_verdict": None,
        "narrative_report": None,
        "analysis_complete": None,
        "task": f"Analyse the following claim for misinformation: {CLAIM[:200]}",
        "plan": [],
        "worker_outputs": [],
        "next": "",
        "instructions": "",
        "article_metadata_cache": {},
        "final_output": None,
    }

    best_state = initial_state
    step = 0
    for state in graph.stream(
        initial_state,
        config={"recursion_limit": config.RECURSION_LIMIT},
        stream_mode="values",
    ):
        step += 1
        best_state = state
        nr = state.get("narrative_report")
        ac = state.get("analysis_complete")
        articles = state.get("retrieved_articles", [])
        clusters = state.get("perspective_clusters", [])
        print(f"[step {step}] analysis_complete={ac} | articles={len(articles)} | clusters={len(clusters)} | has_report={nr is not None}")
        # Exit as soon as we have a complete narrative report
        if ac is True and nr is not None:
            print("[EXIT] analysis_complete=True and narrative_report present — stopping early.")
            break

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slug(CLAIM)
    out_path = OUTPUT_DIR / f"revamped_{slug}_real_{ts}.json"
    out_path.write_text(
        json.dumps(best_state, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n[SAVED] {out_path}")

    # Quick summary
    nr = best_state.get("narrative_report") or {}
    clusters = best_state.get("perspective_clusters", [])
    articles = best_state.get("retrieved_articles", [])
    buckets = nr.get("time_buckets", [])
    inflections = nr.get("inflection_point_cards", [])
    print(f"\n=== REAL PIPELINE SUMMARY ===")
    print(f"  Articles retrieved : {len(articles)}")
    print(f"  Perspective clusters: {len(clusters)}")
    print(f"  Time buckets       : {len(buckets)}")
    print(f"  Inflection points  : {len(inflections)}")
    print(f"  Ground truth tier  : {nr.get('ground_truth_tier', 'N/A')}")
    print(f"  Primary verdict    : {best_state.get('primary_verdict', 'N/A')}")
    return str(out_path)

if __name__ == "__main__":
    run()
