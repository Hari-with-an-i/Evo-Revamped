"""
Evaluator — quality gate between Phase 2 retrieval and the analysis phase.

Four sequential steps (section 6 of narrative_intelligence_architecture.docx):
  Step 1: Domain-root deduplication   — one article per root domain, highest SBERT relevance wins
  Step 2: SBERT relevance filter      — drop < 0.5 cosine; tag 0.5–0.6 as peripheral
  Step 3: NLI perspective clustering  — extract claims, pairwise NLI, LLM-labelled clusters
  Step 4: Coverage gap detection      — check each claim dimension has ≥1 cluster covering it

Outputs evaluation_ready (bool) and coverage_gaps (list[str]) that the orchestrator
uses to decide whether to loop back for more retrieval or proceed to researcher.
"""
from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import config
from src.llm import get_classification_llm
from src.logger import get_logger, timer
from src.schemas.perspective_cluster import CorroborationLevel, PerspectiveCluster
from src.state import AgentState
from src.utils.sbert import cosine_similarities
from src.utils.nli import score_pairs

log = get_logger(__name__)

# --- Thresholds (match architecture doc) -------------------------------------
RELEVANCE_DROP: float = config.RELEVANCE_DROP
RELEVANCE_PERIPHERAL: float = config.RELEVANCE_PERIPHERAL
NLI_ENTAILMENT_THRESHOLD = 0.6  # entailment score above which two claims share a cluster
MAX_ARTICLES_FOR_NLI: int = config.MAX_ARTICLES_FOR_NLI
COVERAGE_THRESHOLD = 0.45  # SBERT cosine: dimension is "covered" if any key claim ≥ this

# Step 2b — source type classification
PRIMARY_NLI_ENTAILMENT    = 0.55  # slightly lower: primary sources use formal/technical language
PRIMARY_NLI_CONTRADICTION = 0.55
SOURCE_TYPE_BODY_CHARS    = 400   # chars of body sent to source-type classifier
MAX_ARTICLES_FOR_SOURCE_CLASSIFY = 30  # batch cap; chunked if exceeded


# --- LLM (lazy, structured output) ------------------------------------------

class _ArticleClaimsResponse(BaseModel):
    all_claims: list[str] = Field(
        description=(
            "Flat list of claims in strict article order: "
            "claim1_for_article1, claim2_for_article1, claim1_for_article2, claim2_for_article2, ... "
            "Always exactly 2 claims per article, so total length = 2 × number of articles."
        )
    )


class _BatchClusterLabelResponse(BaseModel):
    labels: list[str] = Field(
        description="One 1-2 sentence plain-language description per cluster, in exact order."
    )


SourceType = Literal["primary", "secondary"]


class _SourceTypeResponse(BaseModel):
    source_types: str = Field(
        description=(
            "Space-separated classifications, one token per article in order. "
            "Each token must be exactly 'primary' or 'secondary'. "
            "Example for 3 articles: 'secondary primary secondary'"
        )
    )


_SKIP_TOKENS: frozenset[str] = frozenset({"(no claim)", "(skip)"})

_llm_claims = None
_llm_label = None
_llm_source_type = None


def _get_llm_claims():
    global _llm_claims
    if _llm_claims is None:
        _llm_claims = get_classification_llm().with_structured_output(_ArticleClaimsResponse)
    return _llm_claims


def _get_llm_label():
    global _llm_label
    if _llm_label is None:
        _llm_label = get_classification_llm().with_structured_output(_BatchClusterLabelResponse)
    return _llm_label


def _get_llm_source_type():
    global _llm_source_type
    if _llm_source_type is None:
        _llm_source_type = get_classification_llm().with_structured_output(_SourceTypeResponse)
    return _llm_source_type


# --- Step 1: Domain-root deduplication --------------------------------------

def _domain_root(url: str) -> str:
    netloc = urlparse(url).netloc
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def _dedup_by_domain_root(articles: list[dict], claim: str) -> list[dict]:
    """Keep one article per domain root — the one with highest SBERT cosine to the claim."""
    if not articles:
        return []

    by_root: dict[str, list[dict]] = {}
    for a in articles:
        root = _domain_root(a.get("source_url", ""))
        by_root.setdefault(root, []).append(a)

    kept: list[dict] = []
    for root, group in by_root.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            texts = [(a.get("body") or a.get("title") or "")[:500] for a in group]
            sims = cosine_similarities(claim, texts)
            best = group[max(range(len(group)), key=lambda i: sims[i])]
            kept.append(best)

    return kept


# --- Step 2: SBERT relevance filter -----------------------------------------

def _relevance_filter(articles: list[dict], claim: str) -> tuple[list[dict], list[dict]]:
    """
    Return (relevant, peripheral).
    relevant: cosine ≥ RELEVANCE_DROP, peripheral tagged in returned list.
    peripheral: cosine in [RELEVANCE_DROP, RELEVANCE_PERIPHERAL) — included but flagged.
    Articles below RELEVANCE_DROP are dropped entirely.
    """
    if not articles:
        return [], []

    texts = [(a.get("body") or a.get("title") or "")[:500] for a in articles]
    sims = cosine_similarities(claim, texts)

    relevant: list[dict] = []
    peripheral: list[dict] = []
    for a, sim in zip(articles, sims):
        if sim < RELEVANCE_DROP:
            continue
        a = dict(a)  # don't mutate state
        a["_relevance_score"] = round(sim, 4)
        if sim < RELEVANCE_PERIPHERAL:
            a["_peripheral"] = True
            peripheral.append(a)
        else:
            relevant.append(a)

    return relevant, peripheral


# --- Step 3: NLI perspective clustering -------------------------------------

def _extract_claims(articles: list[dict]) -> list[list[str]]:
    """Extract 2 key factual claims per article via a single batch LLM call.

    Skips articles with thin bodies (< 80 chars or flagged as _thin_content).
    Returns a list of [claim1, claim2] pairs aligned to the input article list.
    """
    _THIN = 80
    # Pre-fill all slots with skip placeholders
    result: list[list[str]] = [["(skip)", "(skip)"] for _ in articles]

    slim: list[tuple[int, dict]] = [
        (i, a) for i, a in enumerate(articles)
        if len(a.get("body") or "") >= _THIN and not a.get("_thin_content")
    ]
    if not slim:
        return result

    def _sanitize(text: str) -> str:
        return (
            text
            .replace("‘", "'").replace("’", "'")   # curly single quotes
            .replace("“", '"').replace("”", '"')   # curly double quotes
            .replace("–", "-").replace("—", "--")  # en/em dash
            .encode("ascii", errors="replace").decode("ascii")
        )

    snippets = []
    for j, (_, a) in enumerate(slim):
        title = _sanitize(a.get("title", ""))
        body = _sanitize((a.get("body") or "")[:600])
        snippets.append(f"Article {j+1}: {title}\n{body}")

    n = len(slim)
    prompt = (
        f"You have {n} article(s) below. For each article extract exactly 2 short factual claims "
        f"(one sentence each). Return all claims as a single flat list in strict order: "
        f"claim1_article1, claim2_article1, claim1_article2, claim2_article2, ... "
        f"Total items in the list must be exactly {n * 2}.\n\n"
        + "\n\n---\n\n".join(snippets)
    )
    with timer(log, "llm_call_claims_extraction", node="evaluator"):
        lm_result: _ArticleClaimsResponse = _get_llm_claims().invoke([
            SystemMessage(content=(
                "You extract factual claims from news articles. "
                "Output ONLY the structured JSON tool call with no prose, explanation, or preamble."
            )),
            HumanMessage(content=prompt),
        ])

    flat = lm_result.all_claims
    for j, (orig_i, _) in enumerate(slim):
        c1 = flat[j * 2]     if j * 2     < len(flat) else "(no claim)"
        c2 = flat[j * 2 + 1] if j * 2 + 1 < len(flat) else "(no claim)"
        result[orig_i] = [c1, c2]
    return result


def _filter_garbage_clusters(clusters: list[PerspectiveCluster]) -> list[PerspectiveCluster]:
    """Remove clusters whose key_claims are entirely skip/no-claim placeholders."""
    kept = []
    for c in clusters:
        real = [cl for cl in c.key_claims if cl.strip().lower() not in _SKIP_TOKENS]
        if real:
            kept.append(c.model_copy(update={"key_claims": real}))
    return kept


def _build_clusters(articles: list[dict], claims_per_article: list[list[str]]) -> list[PerspectiveCluster]:
    """
    Build perspective clusters via connected-components on NLI entailment edges.
    Each cluster gets a plain-language label via one LLM call.
    """
    # Flatten: (article_idx, claim_str) for all claims — skip placeholder tokens
    flat: list[tuple[int, str]] = []
    for art_idx, claims in enumerate(claims_per_article):
        for c in claims:
            if c.strip().lower() not in _SKIP_TOKENS:
                flat.append((art_idx, c))

    n = len(flat)
    if n == 0:
        return []

    # Pairwise NLI — only upper triangle to avoid redundant calls
    pairs: list[tuple[str, str]] = []
    pair_indices: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((flat[i][1], flat[j][1]))
            pair_indices.append((i, j))

    scores = score_pairs(pairs)

    # Build adjacency (entailment in either direction)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for (i, j), score_dict in zip(pair_indices, scores):
        if score_dict.get("entailment", 0) >= NLI_ENTAILMENT_THRESHOLD:
            adj[i].add(j)
            adj[j].add(i)

    # Connected components
    visited = set()
    components: list[list[int]] = []
    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adj[node] - visited)
        components.append(component)

    # Build PerspectiveCluster per component
    component_prompts = []
    
    for component in components:
        component_claims = [flat[i][1] for i in component]
        component_prompts.append(
            f"Group claims:\n" + "\n".join(f"- {c}" for c in component_claims[:6])
        )

    cluster_labels = []
    if component_prompts:
        batch_prompt = "\n\n---\n\n".join(
            f"Cluster {i+1} {prompt}" for i, prompt in enumerate(component_prompts)
        )
        with timer(log, "llm_call_cluster_labels_batch", node="evaluator"):
            try:
                label_resp: _BatchClusterLabelResponse = _get_llm_label().invoke([
                    SystemMessage(content=(
                        "You label perspective clusters for a misinformation analysis system. "
                        "Given several clusters of claims, provide exactly one 1-2 sentence "
                        "plain-language description per cluster, in exact order."
                    )),
                    HumanMessage(content=batch_prompt),
                ])
                cluster_labels = label_resp.labels
                if len(cluster_labels) < len(components):
                    cluster_labels.extend(["(cluster)"] * (len(components) - len(cluster_labels)))
            except Exception:
                log.warning("batch cluster labeling failed")
                cluster_labels = ["(cluster)"] * len(components)

    clusters: list[PerspectiveCluster] = []
    for idx, component in enumerate(components):
        art_indices = list({flat[i][0] for i in component})
        component_claims = [flat[i][1] for i in component]
        component_articles = [articles[i] for i in art_indices]

        roots = list({_domain_root(a.get("source_url", "")) for a in component_articles})
        article_ids = [a.get("id", "") or _domain_root(a.get("source_url", "")) for a in component_articles]
        corr_count = len(roots)
        if corr_count <= 2:
            level: CorroborationLevel = "unverified"
        elif corr_count <= 4:
            level = "partial"
        else:
            level = "corroborated"

        clusters.append(PerspectiveCluster(
            label=cluster_labels[idx] if idx < len(cluster_labels) else "(cluster)",
            key_claims=component_claims[:10],
            article_ids=article_ids,
            domain_roots=roots,
            corroboration_count=corr_count,
            corroboration_level=level,
        ))

    return _filter_garbage_clusters(clusters)


# --- Step 4: Coverage gap detection -----------------------------------------

def _detect_gaps(dimensions: list[str], clusters: list[PerspectiveCluster]) -> list[str]:
    """Return dimensions not covered by any cluster's key claims (SBERT cosine < COVERAGE_THRESHOLD)."""
    if not dimensions:
        return []
    if not clusters:
        return list(dimensions)

    all_claims = [c for cluster in clusters for c in cluster.key_claims]
    if not all_claims:
        return list(dimensions)

    gaps: list[str] = []
    for dim in dimensions:
        sims = cosine_similarities(dim, all_claims)
        if not sims or max(sims) < COVERAGE_THRESHOLD:
            gaps.append(dim)
    return gaps


# --- Step 2b: Source type classification ------------------------------------

def _classify_source_types(articles: list[dict]) -> list[dict]:
    """Classify each article as 'primary' or 'secondary' via a single batched LLM call.

    Returns a new list of dicts with '_source_type' injected; originals are not mutated.
    """
    if not articles:
        return []

    system_msg = SystemMessage(content=(
        "You are a source-type classifier for a fact-checking pipeline. "
        "Classify each article as 'primary' or 'secondary'.\n"
        "PRIMARY: peer-reviewed academic paper, official government record or dataset, "
        "court filing, regulatory filing, verified primary data release, clinical trial results.\n"
        "SECONDARY: news article, editorial, opinion piece, blog post, forum discussion, "
        "social media amplification, derivative analysis, commentary on primary sources.\n"
        "When uncertain, default to 'secondary'."
    ))

    all_labels: list[SourceType] = []
    chunk_size = MAX_ARTICLES_FOR_SOURCE_CLASSIFY
    for chunk_start in range(0, len(articles), chunk_size):
        chunk = articles[chunk_start : chunk_start + chunk_size]
        snippets = []
        for i, a in enumerate(chunk):
            url = a.get("source_url", "")
            name = a.get("source_name", "")
            body_preview = (a.get("body") or a.get("title") or "")[:SOURCE_TYPE_BODY_CHARS]
            snippets.append(f"Article {i + 1}:\nURL: {url}\nSource: {name}\n{body_preview}")

        prompt = (
            "Classify each article below as 'primary' or 'secondary'. "
            "Return exactly one label per article, in order.\n\n"
            + "\n\n---\n\n".join(snippets)
        )
        with timer(log, "llm_call_source_classify", node="evaluator"):
            result: _SourceTypeResponse = _get_llm_source_type().invoke([
                system_msg,
                HumanMessage(content=prompt),
            ])
        # Parse flat string back to per-article labels; validate and pad
        raw_tokens = result.source_types.split()
        chunk_labels: list[SourceType] = [
            t if t in ("primary", "secondary") else "secondary"
            for t in raw_tokens
        ]
        while len(chunk_labels) < len(chunk):
            chunk_labels.append("secondary")
        all_labels.extend(chunk_labels[:len(chunk)])

    tagged: list[dict] = []
    for a, label in zip(articles, all_labels):
        tagged_a = dict(a)
        tagged_a["_source_type"] = label
        tagged.append(tagged_a)
    return tagged


# --- Step 3a: Primary NLI verdict -------------------------------------------

def _primary_nli_verdict(primary_articles: list[dict], claim: str) -> str:
    """Return verdict from primary sources only: 'supported'|'contradicted'|'silent'|'none'."""
    if not primary_articles:
        return "none"

    pairs = [
        ((a.get("body") or a.get("title") or "")[:1200], claim)
        for a in primary_articles
    ]
    with timer(log, "nli_primary_track", node="evaluator"):
        scores = score_pairs(pairs)

    statuses: list[str] = []
    for score_dict in scores:
        if score_dict.get("entailment", 0) >= PRIMARY_NLI_ENTAILMENT:
            statuses.append("supports")
        elif score_dict.get("contradiction", 0) >= PRIMARY_NLI_CONTRADICTION:
            statuses.append("contradicts")
        else:
            statuses.append("silent")

    if "contradicts" in statuses:
        return "contradicted"
    if "supports" in statuses:
        return "supported"
    return "silent"


# --- Step 3b: Secondary narrative clustering --------------------------------

def _build_secondary_clusters(
    secondary_articles: list[dict],
    claims_per_article: list[list[str]],
) -> list[PerspectiveCluster]:
    """Thin wrapper around _build_clusters restricted to secondary-tagged articles."""
    return _build_clusters(secondary_articles, claims_per_article)


# --- Node entry point -------------------------------------------------------

def evaluator_node(state: AgentState) -> dict:
    """Quality gate: dedup, filter, cluster, check coverage gaps."""
    articles_in = state.get("retrieved_articles", [])
    claim = state.get("claim", "")
    meta_cache: dict = dict(state.get("article_metadata_cache") or {})
    context_summary = state.get("context_summary") or {}
    raw_dims = context_summary.get("claim_dimensions", "")
    dimensions: list[str] = (
        [d.strip() for d in raw_dims.split("|") if d.strip()]
        if isinstance(raw_dims, str)
        else list(raw_dims)
    )

    log.info("evaluator entered", extra={"article_count": len(articles_in), "dimensions": dimensions})

    if not articles_in:
        log.warning("no articles to evaluate — marking ready with no clusters", extra={"node": "evaluator"})
        return {
            "primary_evidence_exists": False,
            "primary_verdict": "none",
            "perspective_clusters": [],
            "coverage_gaps": [],
            "evaluation_ready": True,
            "worker_outputs": ["[evaluator]\nNo articles — evaluation skipped, marking ready."],
        }

    # Step 1 — domain-root dedup
    deduped = _dedup_by_domain_root(articles_in, claim)
    log.info("evaluator step1 dedup", extra={"before": len(articles_in), "after": len(deduped)})

    # Step 2 — SBERT relevance filter
    relevant, peripheral = _relevance_filter(deduped, claim)
    all_relevant = relevant + peripheral
    log.info("evaluator step2 filter", extra={
        "relevant": len(relevant), "peripheral": len(peripheral),
        "dropped": len(deduped) - len(all_relevant),
    })

    if not all_relevant:
        log.warning("all articles dropped by relevance filter", extra={"node": "evaluator"})
        return {
            "primary_evidence_exists": False,
            "primary_verdict": "none",
            "perspective_clusters": [],
            "coverage_gaps": dimensions,
            "evaluation_ready": False,
            "worker_outputs": [
                "[evaluator]\n"
                f"Articles: {len(articles_in)} → deduped: {len(deduped)} → relevant: 0\n"
                "All articles below relevance threshold — triggering re-query."
            ],
        }

    # Step 2b — source type classification (cap pool first, then split)
    # Check meta_cache first; only classify articles not yet seen in a prior loop.
    nli_pool = all_relevant[:MAX_ARTICLES_FOR_NLI]
    untyped = [a for a in nli_pool if a.get("id", a.get("source_url", "")) not in meta_cache]
    cached_count_type = len(nli_pool) - len(untyped)
    newly_typed = _classify_source_types(untyped) if untyped else []
    for art, typed_art in zip(untyped, newly_typed):
        aid = art.get("id", art.get("source_url", ""))
        meta_cache.setdefault(aid, {})["_source_type"] = typed_art.get("_source_type", "secondary")
    typed_pool = []
    for a in nli_pool:
        aid = a.get("id", a.get("source_url", ""))
        enriched = dict(a)
        enriched["_source_type"] = meta_cache.get(aid, {}).get("_source_type", "secondary")
        typed_pool.append(enriched)
    log.info("evaluator step2b source_types", extra={
        "cached": cached_count_type, "classified": len(untyped),
    })
    primary_arts   = [a for a in typed_pool if a.get("_source_type") == "primary"]
    secondary_arts = [a for a in typed_pool if a.get("_source_type") == "secondary"]

    # Step 3a — primary evidence track
    primary_verdict_str     = _primary_nli_verdict(primary_arts, claim)
    primary_evidence_exists = len(primary_arts) > 0
    log.info("evaluator step3a primary", extra={
        "primary_count": len(primary_arts), "verdict": primary_verdict_str,
    })

    # Step 3b — secondary narrative clustering
    # Check meta_cache for cached claims; only extract for new articles.
    if secondary_arts:
        unclaimed = [
            a for a in secondary_arts
            if meta_cache.get(a.get("id", a.get("source_url", "")), {}).get("_cached_claims") is None
        ]
        cached_count_claims = len(secondary_arts) - len(unclaimed)
        if unclaimed:
            new_claims = _extract_claims(unclaimed)
            for art, claims in zip(unclaimed, new_claims):
                aid = art.get("id", art.get("source_url", ""))
                meta_cache.setdefault(aid, {})["_cached_claims"] = claims
        sec_claims = [
            meta_cache.get(a.get("id", a.get("source_url", "")), {}).get("_cached_claims", [["(no claim)", "(no claim)"]])
            for a in secondary_arts
        ]
        clusters = _build_secondary_clusters(secondary_arts, sec_claims)
        log.info("evaluator step3b claims", extra={
            "cached": cached_count_claims, "extracted": len(unclaimed),
        })
    else:
        clusters = []
    log.info("evaluator step3b secondary_clusters", extra={"cluster_count": len(clusters)})

    # Step 4 — coverage gap detection (secondary clusters only — measures narrative breadth)
    gaps = _detect_gaps(dimensions, clusters)
    ready = len(gaps) == 0
    log.info("evaluator step4 coverage", extra={"gaps": gaps, "ready": ready})

    return {
        "primary_evidence_exists": primary_evidence_exists,
        "primary_verdict":         primary_verdict_str,
        "perspective_clusters":    [c.model_dump() for c in clusters],
        "coverage_gaps":           gaps,
        "evaluation_ready":        ready,
        "article_metadata_cache":  meta_cache,
        "worker_outputs": [
            f"[evaluator]\n"
            f"Articles: {len(articles_in)} → deduped: {len(deduped)} → relevant: {len(all_relevant)}\n"
            f"Primary: {len(primary_arts)} ({primary_verdict_str}) | "
            f"Secondary clusters: {len(clusters)} | Gaps: {gaps if gaps else 'none'} | Ready: {ready}"
        ],
    }
