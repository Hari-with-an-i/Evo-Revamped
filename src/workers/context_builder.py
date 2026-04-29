"""
ContextBuilderWorker — Phase 1, Step 2

Reads the broad-context articles already in state and produces a ContextSummary:
who, what, when, competing_narratives, claim_dimensions.

This runs immediately after broad_context_fetch via a direct graph edge
(not orchestrator-routed), so its output is available when the orchestrator
next wakes for MODE B query planning.
"""
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import config
from src.llm import get_synthesis_llm
from src.logger import get_logger, timer
from src.schemas import Article, ContextSummary
from src.state import AgentState

log = get_logger(__name__)

_SYSTEM = """You are a context analysis specialist for a fact-checking system.

Read the provided articles and the original claim, then produce a structured ContextSummary.

- who:                  Key actors or persons directly involved in the claim.
- what:                 The core event or assertion being made.
- when:                 Inferred time window (ISO range or natural language). Use "unknown" if unclear.
- relevant_time_periods: A JSON-formatted string representing an array of 3-5 distinct historical periods/phases spanning the narrative's lifecycle. Each object must have start_date, end_date (YYYY-MM-DD), and a brief phase_description. Escape quotes properly since this is a string field.
- competing_narratives: Pipe-separated alternative framings or counter-claims present across the articles.
                        Example: "WHO denies link|Telecom industry rejects claim|Blogs assert 5G activated virus"
                        Empty string if no competing narratives found.
- claim_dimensions:     Pipe-separated distinct verifiable sub-questions raised by the claim (2–5).
                        Example: "causal mechanism|timeline accuracy|source credibility|geographical scope"

Be factual. Only infer from what the articles and claim contain — do not fabricate.
"""


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_synthesis_llm().with_structured_output(ContextSummary)
    return _llm


def context_builder_node(state: AgentState) -> dict:
    """Synthesize broad articles into a ContextSummary for the orchestrator."""
    claim = state.get("claim", "")
    articles_raw = state.get("retrieved_articles", [])

    log.info("context_builder entered", extra={"article_count": len(articles_raw)})

    if not articles_raw:
        log.warning("no articles — minimal summary", extra={"node": "context_builder"})
        summary = ContextSummary(
            who="unknown",
            what=claim or "unknown",
            when=state.get("timeframe") or "unknown",
            competing_narratives="",
            claim_dimensions="general claim verification",
        )
        return {
            "context_summary": summary.model_dump(),
            "worker_outputs": ["[context_builder]\nNo broad articles — minimal context generated from claim alone."],
        }

    articles = [Article(**a) for a in articles_raw[:4]]
    articles_text = "\n\n".join(a.to_retrieval_context() for a in articles)

    prompt = (
        f"Claim: {claim}\n\n"
        f"--- ARTICLES ---\n{articles_text}\n--- END ARTICLES ---"
    )

    with timer(log, "llm_call", node="context_builder"):
        summary: ContextSummary = _get_llm().invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])

    dimensions = [d.strip() for d in summary.claim_dimensions.split("|") if d.strip()]
    narratives = [n.strip() for n in summary.competing_narratives.split("|") if n.strip()]
    log.info("context_builder done", extra={
        "dimensions": dimensions,
        "narratives": len(narratives),
    })

    import json
    summary_dict = summary.model_dump()
    try:
        if isinstance(summary_dict.get("relevant_time_periods"), str):
            summary_dict["relevant_time_periods"] = json.loads(summary_dict["relevant_time_periods"])
    except Exception as e:
        log.warning("failed to parse relevant_time_periods JSON string", extra={"error": str(e)})
        summary_dict["relevant_time_periods"] = []

    return {
        "context_summary": summary_dict,
        "worker_outputs": [
            f"[context_builder]\n"
            f"Dimensions: {', '.join(dimensions)}\n"
            f"Time window: {summary.when}\n"
            f"Competing narratives: {len(narratives)}"
        ],
    }
