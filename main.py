from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from src.config import config
from src.graphs import build_graph


def run(raw_input: str, input_type: str = "claim") -> dict:
    """
    Run the full misinformation analysis pipeline.

    Args:
        raw_input:  A direct claim or raw forum/article text.
        input_type: "claim" or "raw_text".

    Returns:
        The final state dict containing retrieved_articles, worker_outputs, etc.
    """
    graph = build_graph()
    initial_state = {
        "messages": [HumanMessage(content=raw_input)],
        # Input
        "raw_input": raw_input,
        "input_type": input_type,
        # Parsed — orchestrator fills these on first call (MODE A)
        "claim": "",
        "entities": [],
        "timeframe": None,
        "claim_complexity": None,
        # Phase 1 context
        "context_summary": None,
        # Retrieval corpus
        "retrieved_articles": [],
        "anchor_articles": [],
        # Phase 2 targeting
        "targeted_queries": [],
        "retrieval_loop_count": 0,
        "retrieval_complete": False,
        # Evaluation
        "perspective_clusters": [],
        "coverage_gaps": [],
        "evaluation_ready": None,
        "primary_evidence_exists": None,
        "primary_verdict": None,
        # Narrative analysis
        "narrative_report": None,
        "analysis_complete": None,
        # Orchestration
        "task": f"Analyse the following {input_type} for misinformation: {raw_input[:200]}",
        "plan": [],
        "worker_outputs": [],
        "next": "",
        "instructions": "",
        "article_metadata_cache": {},
        "final_output": None,
    }

    last_state = initial_state
    try:
        for state in graph.stream(
            initial_state,
            config={"recursion_limit": config.RECURSION_LIMIT},
            stream_mode="values",
        ):
            last_state = state
    except GraphRecursionError:
        print("[WARNING] Recursion limit reached — returning partial state")

    return last_state


if __name__ == "__main__":
    import sys

    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "The WHO confirmed that 5G towers spread COVID-19 in densely populated cities."
    )
    itype = "claim"

    result = run(raw, itype)

    final = result.get("final_output")
    if final:
        print("\n=== REPORT ===")
        print(final)
    else:
        print("\n=== WORKER OUTPUTS ===")
        for output in result.get("worker_outputs", []):
            print(output)
            print("---")

    articles = result.get("retrieved_articles", [])
    print(f"\n=== CORPUS: {len(articles)} articles ===")
    for a in articles[:5]:
        print(f"  [{a.get('outlet_type')}] {a.get('source_name')} — {a.get('title', '')[:80]}")
