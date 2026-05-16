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
import json
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
You have a context_summary with who/what/when/dimensions and relevant_time_periods. 
Generate targeted queries for EACH relevant time period, setting start_date and end_date on the query to match the time period.
Assign each query to one tool:
  "tavily"            — full-text, recent web results
  "gdelt_commoncrawl" — date-filtered news corpus (GDELT) + web archive (Common Crawl/BigQuery)
  "scholar_wiki"      — Wikipedia anchoring (stored separately; used for grounding only, not cited in the final report)

Cover different claim dimensions with different tools across the time periods. Be specific.
Set next = "tavily_targeted" (sequential Phase 2 entry point).
For any claim with a specific year, event, or named period, you MUST generate at least 2 queries with tool="gdelt_commoncrawl" and explicit start_date/end_date matching the claim's timeframe. Tavily is for current/recent coverage only (no date bounds). If the claim is entirely about events in the last 3 months, all queries may use tavily.
Set retrieval_complete to the boolean false (not the string "false").

Set new_targeted_queries_json to a valid JSON array string. Example:
[{"query":"vaccine safety clinical trials","tool":"gdelt_commoncrawl","dimension":"causal mechanism","start_date":"2020-01-01","end_date":"2022-12-31"},{"query":"vaccine mandate policy debate","tool":"tavily","dimension":"political framing","start_date":"","end_date":""}]
All four string fields (query, tool, dimension, start_date, end_date) must be present in each object. Use empty string for missing dates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE C — LOOP EVALUATION  (after Phase 2 workers or evaluator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The evaluator is the quality gate. Use evaluation_ready and coverage_gaps from context.

After Phase 2 retrieval workers have run:
  • evaluation_ready is None → set next = "evaluator"  (must run quality assessment first)

After evaluator + analyst have run (analysis_complete will be set):
  • evaluation_ready = False AND loop_count < MAX_LOOPS
      → generate new_targeted_queries_json for the specific coverage_gaps only
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
    # MODE B/C outputs — queries encoded as a JSON string to stay within 8B tool-call limits.
    # Format: '[{"query":"...","tool":"tavily","dimension":"...","start_date":"...","end_date":"..."},...]'
    # Empty string means no new queries this turn.
    new_targeted_queries_json: str = Field(
        default="",
        description=(
            "JSON array string of targeted queries for Phase 2. MODE B/C only. "
            'Each item: {"query":"...","tool":"tavily|gdelt_commoncrawl|scholar_wiki",'
            '"dimension":"...","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}. '
            "Empty string when not generating new queries."
        ),
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
            max_tokens=900,
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

        when_str = cs.get('when', '')
        when_note = (
            f"\n  NOTE: 'When' contains a historical date range — use this as "
            f"start_date/end_date for gdelt_commoncrawl queries to retrieve archived news from that period."
            if any(str(y) in when_str for y in range(2018, 2026))
            else ""
        )
        context_parts.append(
            f"Context Summary:\n"
            f"  Who: {cs.get('who', '')}\n"
            f"  What: {cs.get('what', '')}\n"
            f"  When: {when_str}{when_note}\n"
            f"  Timeframe from claim: {state.get('timeframe') or 'not specified'}\n"
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
            truncated = [o[:400] + ("…" if len(o) > 400 else "") for o in recent]
            context_parts.append("Recent worker outputs:\n" + "\n---\n".join(truncated))

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
        # Guard 0: drain any unconsumed GDELT queries before running the evaluator.
        # When MODE B generated gdelt_commoncrawl queries they sit in targeted_queries
        # until gdelt_commoncrawl_targeted picks them up. If Tavily just ran but GDELT
        # hasn't yet, route to gdelt_commoncrawl_targeted first.
        gdelt_queries_pending = any(
            q.get("tool") == "gdelt_commoncrawl"
            for q in state.get("targeted_queries", [])
        )
        gdelt_ran = any(
            o.startswith("[gdelt_commoncrawl_targeted]")
            for o in state.get("worker_outputs", [])
        )
        if gdelt_queries_pending and not gdelt_ran and state.get("evaluation_ready") is None:
            log.info("orchestrator guard: GDELT queries pending — routing to gdelt_commoncrawl_targeted")
            decision.next = "gdelt_commoncrawl_targeted"

        # Guard 1 (highest): evaluator must run before any Phase 3 action
        elif state.get("evaluation_ready") is None and decision.next != "evaluator":
            log.warning("orchestrator guard: forcing evaluator (evaluation_ready is None)")
            decision.next = "evaluator"

        # Guard 2: retrieval is complete and analyst has run — advance the pipeline forward.
        # Only fires once retrieval_complete=True so GDELT loop passes are not blocked.
        # Determine the correct next step by inspecting which nodes have already run.
        elif state.get("retrieval_complete") is True and decision.next not in (
            "researcher", "writer", "FINISH"
        ):
            outputs = state.get("worker_outputs", [])
            researcher_ran = any(o.startswith("[researcher]") for o in outputs)
            writer_ran = any(o.startswith("[writer]") for o in outputs)
            if writer_ran:
                forced_next: WorkerName = "FINISH"
            elif researcher_ran:
                forced_next = "writer"
            else:
                forced_next = "researcher"
            log.warning(
                "orchestrator guard: analysis_complete=True, next=%s — forcing %s",
                decision.next, forced_next,
            )
            decision.next = forced_next
            decision.retrieval_complete = True

        # Guard 3: loop ceiling hit → force advance regardless of LLM
        elif (
            state.get("evaluation_ready") is not None
            and loop_count >= config.MAX_RETRIEVAL_LOOPS
            and decision.next not in ("researcher", "writer", "FINISH")
        ):
            log.warning(
                "orchestrator guard: ceiling %d/%d — forcing researcher",
                loop_count, config.MAX_RETRIEVAL_LOOPS,
            )
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

    # MODE B/C: parse targeted queries from JSON string and write to state.
    # The JSON-string field sidesteps 8B model limitations with nested list schemas.
    new_queries: list[TargetedQuery] = []
    if decision.new_targeted_queries_json.strip():
        try:
            raw_qs = json.loads(decision.new_targeted_queries_json)
            if isinstance(raw_qs, list):
                for item in raw_qs:
                    if isinstance(item, dict) and item.get("query") and item.get("tool"):
                        try:
                            new_queries.append(TargetedQuery(**item))
                        except Exception:
                            pass
        except (json.JSONDecodeError, Exception):
            log.warning("orchestrator: failed to parse new_targeted_queries_json")

    if new_queries:
        update["targeted_queries"] = [q.model_dump() for q in new_queries]
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
