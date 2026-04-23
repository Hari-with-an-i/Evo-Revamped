from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_synthesis_llm
from src.logger import get_logger, timer
from src.schemas import Article
from src.state import AgentState

log = get_logger(__name__)

_SYSTEM = (
    "You are a meticulous research specialist. "
    "Given a claim and retrieved evidence, analyse the evidence critically "
    "and return a concise, well-structured research brief. "
    "Do not call any tools — all evidence has already been retrieved."
)


def researcher_node(state: AgentState) -> dict:
    claim = state.get("claim") or state.get("raw_input", "")
    articles_raw = state.get("retrieved_articles") or []
    narrative_report = state.get("narrative_report")
    clusters = state.get("perspective_clusters") or []
    coverage_gaps = state.get("coverage_gaps") or []
    primary_verdict = state.get("primary_verdict")

    # Build evidence digest from top articles
    articles = [Article(**a) for a in articles_raw[:10]]
    evidence_text = "\n\n".join(a.to_retrieval_context() for a in articles)

    parts = [f"Claim: {claim}"]

    if primary_verdict:
        parts.append(f"Primary source verdict: {primary_verdict}")

    if clusters:
        cluster_lines = "\n".join(
            f"  - {c.get('label', 'unknown')} ({c.get('corroboration_level', '')}): {c.get('description', '')}"
            for c in clusters[:6]
        )
        parts.append(f"Perspective clusters:\n{cluster_lines}")

    if coverage_gaps:
        parts.append(f"Coverage gaps: {', '.join(coverage_gaps)}")

    if narrative_report:
        summary = narrative_report.get("narrative_intelligence_summary", "")
        if summary:
            parts.append(f"Analyst narrative summary:\n{summary}")

    parts.append(f"Evidence articles:\n{evidence_text}")
    parts.append("Task: Write a concise research brief analysing this claim against the evidence above.")

    prompt = "\n\n".join(parts)

    llm = get_synthesis_llm()
    with timer(log, "llm_call", worker="researcher"):
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])

    log.info("researcher completed", extra={"output_len": len(response.content)})
    return {"worker_outputs": [f"[researcher]\n{response.content}"]}
