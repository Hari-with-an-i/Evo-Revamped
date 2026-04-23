"""
Central Orchestrator

Single node, single LLM call per invocation, three operating modes determined by state:

  MODE A — Input parsing (first call, claim is empty)
    Reads:  raw_input, input_type
    Writes: claim, entities, timeframe, claim_complexity
    Routes: → broad_context_fetch

  MODE B — Query planning (after Phase 1 context is ready)
    Reads:  claim, context_summary
    Writes: targeted_queries (appended), retrieval_loop_count
    Routes: → tavily_targeted (first Phase 2 worker)

  MODE C — Loop evaluation / pipeline advance
    Reads:  retrieved_articles count, retrieval_loop_count, worker_outputs
    Writes: retrieval_complete (when done), targeted_queries (if looping)
    Routes: → next Phase 2 worker, or researcher, or writer, or FINISH
"""
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from src.config import config
from src.logger import get_logger, timer
from src.schemas import TargetedQuery
from src.state import AgentState

log = get_logger(__name__)

WorkerName = Literal[
    "broad_context_fetch",
    "tavily_targeted",
    "gdelt_commoncrawl_targeted",
    "scholar_wiki_targeted",
    "evaluator",
    "researcher",
    "writer",
    "FINISH",
]

# context_builder is intentionally absent from WorkerName —
# the graph wires it as an unconditional edge from broad_context_fetch.

SYSTEM_PROMPT = """You are the central orchestrator for a multi-phase misinformation analysis system.

You operate in one of three modes, indicated in the context you receive each turn.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE A — INPUT PARSING  (claim is empty)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read raw_input and input_type. Populate:
  parsed_claim       — one declarative, falsifiable sentence; strip slang/URLs/emojis
  parsed_entities    — comma-separated key named people, orgs, locations (max 8, only what's in the text); empty string if none
  parsed_timeframe   — ISO date or range if present, else empty string
  claim_complexity   — "simple" if claim is self-contained and unambiguous;
                       "complex" if it involves multiple actors, causal chains, or disputed facts
Always set next = "broad_context_fetch".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE B — QUERY PLANNING  (Phase 1 complete, context ready)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have a context_summary with who/what/when/dimensions. Generate 3–5 targeted queries,
each assigned to one tool:
  "tavily"            — full-text, recent web results
  "gdelt_commoncrawl" — date-filtered news corpus (GDELT) + web archive (Common Crawl/BigQuery)
  "scholar_wiki"      — Wikipedia anchoring (stored separately; used for grounding only, not cited in the final report)

Cover different claim dimensions with different tools. Be specific — not generic.
Set next = "tavily_targeted" (sequential Phase 2 entry point).
Prefer "gdelt_commoncrawl" for queries that need date-filtered or archival coverage.
Set retrieval_complete to the boolean false (not the string "false").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE C — LOOP EVALUATION  (after Phase 2 workers or evaluator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The evaluator is the quality gate. Use evaluation_ready and coverage_gaps from context.

After Phase 2 retrieval workers have run:
  • evaluation_ready is None → set next = "evaluator"  (must run quality assessment first)

After evaluator + analyst have run (analysis_complete will be set):
  • evaluation_ready = False AND loop_count < MAX_LOOPS
      → generate new_targeted_queries for the specific coverage_gaps only
      → set next = appropriate Phase 2 worker (tavily_targeted for fresh web coverage,
        gdelt_commoncrawl_targeted for archival/temporal gaps)
  • evaluation_ready = True OR loop_count >= MAX_LOOPS
      → set retrieval_complete to the boolean true (not the string "true"), next = "researcher"
      → if loop_count >= MAX_LOOPS but evaluation_ready = False, still proceed — gaps will be
        flagged in output
  • analysis_complete = False (analyst short-circuited due to incomplete evaluation)
      → treat as evaluation_ready = False and continue loop normally
  Note: analyst failure is non-blocking — if analysis_complete is False but
  evaluation_ready = True, still route to researcher.

After researcher is done: next = "writer".
After writer is done:     next = "FINISH".

Primary evidence context (when available):
  primary_evidence_exists = True means ≥1 primary source (peer-reviewed paper,
  government record, court filing) was found in the corpus.
  primary_verdict indicates whether primary sources support, contradict, or are
  silent on the claim. A primary_verdict of "contradicted" is a strong signal —
  route to researcher even if coverage_gaps remain; the contradiction is the finding.
  A primary_verdict of "none" means no primary sources found — treat corpus as
  secondary-only and weight coverage_gaps more heavily.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Be concise and directive. Do not do the work yourself — delegate it.
Only populate fields relevant to the active mode.
"""


class OrchestratorDecision(BaseModel):
    reasoning: str = Field(default="", description="One sentence explaining this routing decision.")
    next: WorkerName = Field(description="Which worker to call next, or 'FINISH'.")
    instructions: str = Field(
        default="",
        description="Specific instructions for the chosen worker. Empty when next='FINISH'.",
    )
    updated_plan: str = Field(
        default="",
        description="Comma-separated updated plan steps. Only include when the plan changes. Leave empty otherwise.",
    )
    # MODE A outputs
    parsed_claim: str = Field(default="", description="Normalized claim. MODE A only.")
    parsed_entities: str = Field(
        default="",
        description="Comma-separated key named entities (people, orgs, locations). MODE A only. Empty string when not in MODE A.",
    )
    parsed_timeframe: str = Field(default="", description="ISO date or range if present, else empty string. MODE A only.")
    claim_complexity: str = Field(default="", description="'simple' or 'complex'. MODE A only.")
    # MODE B/C outputs
    new_targeted_queries: list[TargetedQuery] = Field(
        default_factory=list,
        description="Targeted queries for Phase 2. MODE B/C only.",
    )
    retrieval_complete: Any = Field(
        default=False,
        description="Boolean. Set true when corpus is sufficient. MODE C only.",
    )

    @field_validator("retrieval_complete", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=config.MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            max_tokens=600,
        ).with_structured_output(OrchestratorDecision)
    return _llm


def orchestrator_node(state: AgentState) -> dict:
    """
    Single orchestrator node. Determines its own operating mode from state
    and makes all routing decisions for the pipeline.
    """
    claim = state.get("claim", "")
    context_summary = state.get("context_summary")
    retrieval_complete = state.get("retrieval_complete", False)
    loop_count = state.get("retrieval_loop_count", 0)
    articles = state.get("retrieved_articles", [])

    # Determine mode label for logging
    if not claim.strip():
        _mode = "A"
    elif context_summary is None:
        _mode = "A_recovery"
    elif not retrieval_complete and loop_count == 0:
        _mode = "B"
    else:
        _mode = "C"

    log.info("orchestrator entered", extra={
        "mode": _mode, "claim": claim[:80], "loop": loop_count,
        "article_count": len(articles),
    })

    # --- Build mode-specific context ---
    context_parts: list[str] = [f"## Task\n{state.get('task', '')}"]

    if not claim.strip():
        # MODE A
        context_parts.append("## ACTIVE MODE: A — INPUT PARSING")
        context_parts.append(f"raw_input:\n{state.get('raw_input', '')}")
        context_parts.append(f"input_type: {state.get('input_type', 'claim')}")

    elif context_summary is None:
        # This state is only reachable if context_builder hasn't run yet.
        # Under correct graph wiring this shouldn't occur, but handle gracefully.
        context_parts.append("## ACTIVE MODE: A (recovery) — claim set but no context yet")
        context_parts.append(f"Claim: {claim}")
        context_parts.append("Route to broad_context_fetch to rebuild context.")

    elif not retrieval_complete and loop_count == 0:
        # MODE B — first time after Phase 1
        context_parts.append("## ACTIVE MODE: B — QUERY PLANNING")
        context_parts.append(f"Claim: {claim}")
        if state.get("entities"):
            context_parts.append(f"Entities: {', '.join(state['entities'])}")
        if state.get("timeframe"):
            context_parts.append(f"Timeframe: {state['timeframe']}")
        cs = context_summary
        raw_dims = cs.get('claim_dimensions', '')
        dimensions = [d.strip() for d in raw_dims.split("|") if d.strip()] if isinstance(raw_dims, str) else raw_dims
        raw_narr = cs.get('competing_narratives', '')
        narratives = [n.strip() for n in raw_narr.split("|") if n.strip()] if isinstance(raw_narr, str) else raw_narr
        context_parts.append(
            f"Context Summary:\n"
            f"  Who: {cs.get('who', '')}\n"
            f"  What: {cs.get('what', '')}\n"
            f"  When: {cs.get('when', '')}\n"
            f"  Dimensions: {', '.join(dimensions)}\n"
            f"  Competing narratives: {len(narratives)}"
        )
        context_parts.append(
            f"MIN_ARTICLES target: {config.MIN_ARTICLES} | "
            f"MAX_LOOPS: {config.MAX_RETRIEVAL_LOOPS}"
        )

    else:
        # MODE C — loop evaluation
        context_parts.append("## ACTIVE MODE: C — LOOP EVALUATION")
        context_parts.append(f"Claim: {claim}")
        anchor_count = len(state.get("anchor_articles", []))
        context_parts.append(
            f"Evidence articles retrieved so far: {len(articles)} "
            f"(target: {config.MIN_ARTICLES}) — Wikipedia anchor pages ({anchor_count}) excluded from count"
        )
        context_parts.append(
            f"Retrieval loop: {loop_count} / {config.MAX_RETRIEVAL_LOOPS}"
        )
        evaluation_ready = state.get("evaluation_ready")
        analysis_complete = state.get("analysis_complete")
        coverage_gaps = state.get("coverage_gaps", [])
        context_parts.append(f"evaluation_ready: {evaluation_ready}")
        context_parts.append(f"analysis_complete: {analysis_complete}")
        if coverage_gaps:
            context_parts.append(f"coverage_gaps: {', '.join(coverage_gaps)}")
        primary_verdict = state.get("primary_verdict")
        primary_evidence_exists = state.get("primary_evidence_exists")
        if primary_verdict is not None:
            context_parts.append(
                f"primary_evidence_exists: {primary_evidence_exists} | "
                f"primary_verdict: {primary_verdict}"
            )
        if state.get("worker_outputs"):
            recent = state["worker_outputs"][-3:]
            context_parts.append("Recent worker outputs:\n" + "\n---\n".join(recent))

    if state.get("plan"):
        context_parts.append(
            "## Plan\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
        )

    context = "\n\n".join(context_parts)
    # Include conversation history only for MODE A — the raw_input is already in the
    # context block for MODE B/C, so passing the original HumanMessage again is redundant.
    prior_messages = list(state.get("messages", [])) if _mode == "A" else []
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + prior_messages + [
        {"role": "user", "content": context}
    ]

    with timer(log, "llm_call", node="orchestrator", mode=_mode):
        decision: OrchestratorDecision = _get_llm().invoke(messages)

    # Hard guards — override LLM if it ignores critical state
    if _mode == "C":
        if state.get("evaluation_ready") is None and decision.next != "evaluator":
            # Evaluator must always run before researcher — highest priority guard
            decision.next = "evaluator"
        elif (state.get("evaluation_ready") is not None
              and loop_count >= config.MAX_RETRIEVAL_LOOPS
              and not decision.retrieval_complete):
            # Loop ceiling hit after evaluator has run — force through to researcher
            decision.next = "researcher"
            decision.retrieval_complete = True

    log.info("orchestrator routing", extra={"next": decision.next, "reasoning": decision.reasoning})

    update: dict = {
        "next": decision.next,
        "instructions": decision.instructions,
    }

    # MODE A writes
    if not claim.strip() and decision.parsed_claim:
        update["claim"] = decision.parsed_claim
        update["entities"] = [e.strip() for e in decision.parsed_entities.split(",") if e.strip()]
        update["timeframe"] = decision.parsed_timeframe or None
        update["claim_complexity"] = decision.claim_complexity or "complex"

    # MODE B/C: new targeted queries — reset evaluation_ready so the hard guard
    # reliably forces the evaluator after the next retrieval batch completes.
    if decision.new_targeted_queries:
        update["targeted_queries"] = [q.model_dump() for q in decision.new_targeted_queries]
        update["retrieval_loop_count"] = loop_count + 1
        update["evaluation_ready"] = None

    if decision.retrieval_complete:
        update["retrieval_complete"] = True

    if decision.updated_plan:
        update["plan"] = [s.strip() for s in decision.updated_plan.split(",") if s.strip()]

    return update


def route_from_orchestrator(state: AgentState) -> WorkerName:
    """Conditional edge: read the routing decision the orchestrator wrote to state."""
    return state["next"]  # type: ignore[return-value]
