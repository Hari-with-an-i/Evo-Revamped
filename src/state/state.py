import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state flowing through every node in the orchestrator-worker graph.

    --- Conversation ---
    messages:             Conversation history with the user.

    --- Input ---
    raw_input:            Original user-supplied text or claim, unmodified.
    input_type:           Whether the input is a direct claim or raw forum/article text.

    --- Parsed (written by orchestrator MODE A) ---
    claim:                Normalized, single-sentence falsifiable claim.
    entities:             Key named entities extracted from the claim.
    timeframe:            Optional date/range hint from the input.
    claim_complexity:     "simple" or "complex" — governs retrieval depth.

    --- Phase 1 context (written by context_builder) ---
    context_summary:      ContextSummary.model_dump() dict — who/what/when/dimensions.

    --- Retrieval corpus ---
    retrieved_articles:   Evidence corpus — appended by Tavily, GDELT, and Common Crawl workers only.
    anchor_articles:      Wikipedia pages for entity/concept grounding — never included in the report.

    --- Phase 2 targeting (written by orchestrator MODE B/C) ---
    targeted_queries:     List of TargetedQuery dicts — appended each orchestrator loop.
    retrieval_loop_count: How many Phase 2 loops have run; incremented by the orchestrator.
    retrieval_complete:   Set True by orchestrator when corpus is sufficient or max loops reached.

    --- Evaluation (written by evaluator; replace semantics — latest run wins) ---
    perspective_clusters:    NLI-derived perspective clusters from most recent evaluation.
    coverage_gaps:           Claim dimensions not yet covered by any cluster.
    evaluation_ready:        True when coverage is complete; None before first evaluator run.
    primary_evidence_exists: True when ≥1 primary source found; None before first evaluator run.
    primary_verdict:         NLI verdict from primary sources only: "supported"|"contradicted"|"silent"|"none"|None.

    --- Narrative Analysis (written by analyst; replace semantics — latest run wins) ---
    narrative_report:        NarrativeReport.model_dump() dict — None until analyst runs.
    analysis_complete:       True when analyst has produced a report; None before first run.

    --- Orchestration ---
    task:                 High-level task description.
    plan:                 Step-by-step plan produced by the orchestrator.
    worker_outputs:       Accumulated string summaries from all worker calls.
    next:                 Routing decision — worker name or "FINISH".
    instructions:         Instructions the orchestrator passes to the next worker.
    """
    messages: Annotated[list[BaseMessage], add_messages]

    # Input
    raw_input: str
    input_type: Literal["claim", "raw_text"]

    # Parsed
    claim: str
    entities: list[str]
    timeframe: str | None
    claim_complexity: Literal["simple", "complex"] | None

    # Phase 1 context — stored as dict (JSON-serializable for checkpointing)
    context_summary: dict | None

    # Retrieval corpus — evidence articles used in the final report
    retrieved_articles: Annotated[list[dict], operator.add]
    # Anchor articles — Wikipedia only; used for entity/concept grounding, never cited in the report
    anchor_articles: Annotated[list[dict], operator.add]

    # Phase 2 targeting
    targeted_queries: Annotated[list[dict], operator.add]
    retrieval_loop_count: int
    retrieval_complete: bool

    # Evaluation (written by evaluator; replace semantics — latest run wins)
    perspective_clusters: list[dict]       # NLI perspective clusters from most recent evaluation
    coverage_gaps: list[str]               # uncovered claim dimensions from most recent evaluation
    evaluation_ready: bool | None          # None = evaluator not yet run this loop
    primary_evidence_exists: bool | None   # None before first eval run
    primary_verdict: str | None            # "supported"|"contradicted"|"silent"|"none"|None

    # --- Narrative Analysis (written by analyst; replace semantics — latest run wins) ---
    narrative_report: dict | None       # NarrativeReport.model_dump() — None until analyst runs
    analysis_complete: bool | None      # None before first analyst run

    # Orchestration
    task: str
    plan: list[str]
    worker_outputs: Annotated[list[str], operator.add]
    next: str
    instructions: str

    # Per-article metadata cache — keyed by article ID; avoids re-classifying across loops
    # Format: {article_id: {"_source_type": str, "_cached_claims": list[list[str]]}}
    article_metadata_cache: dict

    # Final rendered output — written by writer; never fed back into orchestrator context
    final_output: str | None
