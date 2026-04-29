# Proposed Methodology: Agentic Narrative Intelligence System for Claim Analysis

---

## Context

This methodology describes the design and implementation of an automated, multi-phase agentic pipeline for analyzing claims and producing structured narrative intelligence reports. The system was built to go beyond binary true/false verdicts — instead revealing *how* a narrative evolved over time, *who* was driving it, and *why* shifts occurred. The core motivation was investigative journalism-grade analysis at scale, combining retrieval, semantic reasoning, and temporal analytics into a single end-to-end workflow.

---

## 1. System Overview

The system follows an **orchestrator-worker architecture** built on LangGraph, a graph-based agent framework. A central orchestrator LLM makes routing and planning decisions, while a set of specialized worker modules execute retrieval, evaluation, analysis, and synthesis tasks. The entire pipeline is governed by a shared state object that accumulates evidence and context across phases.

The pipeline is divided into three major phases:
1. **Context Discovery** — parsing the input and gathering initial context
2. **Targeted Multi-Source Retrieval** — iterative, directed evidence gathering
3. **Narrative Intelligence Analysis** — multi-track temporal analysis and report generation

---

## 2. Input Parsing and Claim Normalization

The pipeline accepts a raw natural-language claim or article text as input. The orchestrator LLM (Llama 3.3 70B via Groq) parses this into a normalized claim representation consisting of:

- **Entities** — named persons, organizations, locations, and concepts central to the claim
- **Timeframe** — inferred temporal scope of the assertion
- **Claim complexity** — a signal used to calibrate retrieval depth

This normalization ensures downstream components operate on a clean, structured representation rather than raw, ambiguous text.

---

## 3. Broad Context Discovery

Before targeted retrieval begins, the system executes a fast, broad-context fetch using **Tavily** (a real-time web search API). One to two queries are constructed from the normalized claim and entities and executed without LLM involvement — purely mechanical retrieval.

The resulting articles are fed into the **Context Builder**, which uses a lightweight synthesis LLM (Llama 3.1 8B) to extract:

- **Who, What, When** — the principal actors, core assertion, and time window
- **Competing narratives** — counter-claims and alternative framings found in the articles
- **Claim dimensions** — two to five distinct, verifiable sub-questions that together constitute the full claim
- **Relevant time periods** — three to five historical phases (each with explicit start/end dates and a plain-language description) that capture the lifecycle of the narrative

The time periods are particularly critical: they form the temporal scaffold on which all downstream analysis is anchored.

---

## 4. Targeted Multi-Source Retrieval

Using the context summary, the orchestrator plans a set of **targeted queries**, each assigned to one of three retrieval tools:

### 4.1 Tavily (Recent Web Search)
Full-text, recency-weighted web search. Used for recent developments and broadly accessible coverage.

### 4.2 GDELT + Common Crawl (Archival News Corpus)
- **GDELT** (Global Database of Events, Language, and Tone) is a public, date-filterable global news index. It enables retrieval of articles from specific time windows, essential for historical analysis.
- **Common Crawl** is a web archive queried via Google BigQuery. It provides access to older and less-indexed content, augmenting GDELT coverage.
- Both sources are queried in parallel and results are normalized into a unified article schema.

### 4.3 Wikipedia (Conceptual Grounding)
Wikipedia pages for key entities are fetched as **anchor articles** — used internally to ground entity disambiguation but never cited in the final report.

All retrieved articles are normalized into a standardized schema capturing: title, body text, source URL, publication date, source name, and outlet type (mainstream / state media / independent / unknown).

### 4.4 Iterative Retrieval Loop
The retrieval phase is **iterative** (up to three loops). After each round, the orchestrator evaluates coverage: if key claim dimensions remain unaddressed, additional targeted queries are generated to fill those gaps. This ensures the evidence corpus is both broad and dimensionally complete before analysis begins.

---

## 5. Corpus Evaluation and Quality Filtering

Before analysis, the retrieved corpus passes through a rigorous **evaluation pipeline** with four sequential steps:

### 5.1 Domain-Root Deduplication
Only one article per unique domain root is retained. When multiple articles from the same domain exist, the one with the highest semantic similarity to the claim (measured by SBERT) is kept. This prevents any single outlet from disproportionately influencing the analysis.

### 5.2 Semantic Relevance Filtering (SBERT)
All articles are scored for semantic similarity to the claim using **SBERT** (Sentence-BERT, all-MiniLM-L6-v2). Articles below a threshold of 0.35 cosine similarity are discarded. Articles scoring between 0.35 and 0.50 are retained but tagged as "peripheral."

### 5.3 Perspective Clustering via NLI
This is the methodological centerpiece of the evaluation phase:

1. A classification LLM (Llama 4 Scout 17B) extracts two concrete factual claims per article.
2. All extracted claims are scored pairwise using an **NLI model** (CrossEncoder nli-deberta-v3-small) to measure entailment relationships.
3. Claim pairs with entailment scores ≥ 0.60 are linked; connected components of linked claims form **perspective clusters**.
4. Each cluster is labeled by a batch LLM call with a 1–2 sentence plain-language description.
5. Clusters are assigned a **corroboration level** based on the number of unique domain roots contributing to them: 1–2 sources = "unverified," 3–4 = "partial corroboration," 5+ = "corroborated."

This approach groups articles by the *positions they take* rather than by their source or topic, revealing the actual landscape of competing viewpoints.

### 5.4 Coverage Gap Detection
Each claim dimension is checked for corpus coverage: a dimension is "covered" if at least one perspective cluster has SBERT cosine similarity ≥ 0.45 to that dimension. Uncovered dimensions are flagged as **coverage gaps** and trigger additional retrieval loops if the loop budget permits.

### 5.5 Primary Source Verdict
Articles classified as primary sources (peer-reviewed papers, government records, court filings) are separately evaluated via NLI against the original claim. This produces a **primary source verdict**: supported, contradicted, or silent.

---

## 6. Temporal Bucketing

All articles with valid publication dates are partitioned into **time buckets** using an adaptive algorithm:

- If the context builder provided predefined time periods (the common case), each article is assigned to the closest matching period bucket.
- Otherwise, an adaptive algorithm divides the corpus date range into N equal-width buckets, where N = max(3, article_count ÷ 5), capped at the number of articles.

This bucketing scheme is the temporal foundation for all four analysis tracks described in the next section.

---

## 7. Multi-Track Narrative Intelligence Analysis

The analyst module runs **four parallel analytical tracks** over the time-bucketed corpus, then synthesizes inflection points where multiple tracks signal simultaneously.

### Track 1: Sentiment Analysis
- **Method**: RoBERTa-based sentiment classifier
- **Process**: Entity-relevant sentences are extracted per bucket, scored on a −1.0 to +1.0 scale, and averaged. The delta between consecutive buckets is computed.
- **Output**: A sentiment timeline with per-bucket scores and deltas, indicating emotional tone shifts in media coverage.

### Track 2: Narrative Frame Evolution
- **Method**: TF-IDF + LLM classification
- **Process**: The top 20 TF-IDF terms from each bucket's articles are computed. A single batch LLM call classifies each bucket's dominant frame from: scientific, political, economic, legal, cultural, health, or other.
- **Output**: A frame evolution log showing how the dominant lens of coverage changed over time.

### Track 3: Voice Composition
- **Method**: Perspective cluster mapping
- **Process**: Each article is mapped to its NLI-derived cluster label. Per bucket, the dominant cluster (by article count) is identified. Shifts in dominant cluster between consecutive buckets are flagged.
- **Output**: A voice composition timeline showing which narrative perspective held sway during each period.

### Track 4: External Event Correlation (GDELT)
- **Method**: GDELT event query + LLM plausibility scoring
- **Process**: For "spike buckets" (any single track signal firing — sentiment delta ≥ 0.30, frame change, or voice shift), GDELT is queried for real-world events in a ±3-day window around the bucket midpoint. The LLM selects the most plausibly causal event and assigns a plausibility score (0.0–1.0).
- **Output**: Per-spike event correlations linking narrative shifts to verifiable real-world triggers.

### Inflection Point Detection
An **inflection point** is declared when ≥ 2 tracks fire simultaneously in the same bucket (e.g., sentiment shift + frame change). The LLM then generates a 2–3 sentence characterization of the inflection: what changed, which perspective gained dominance, and why it is analytically significant.

### Ground Truth Classification
Based on cluster corroboration ratios and primary source verdict, each claim is classified into one of three tiers:
- **Verifiable** — corroborated clusters exist and primary sources support the claim
- **Contested** — competing clusters with partial corroboration; no decisive primary evidence
- **Unresolvable** — insufficient evidence or irreconcilable primary source conflict

---

## 8. Synthesis and Report Generation

### Research Brief
A synthesis LLM (Llama 3.1 8B) consumes the narrative report and evidence corpus to produce a concise research brief summarizing key findings.

### Narrative Intelligence Summary
A final call to the 70B orchestrator model generates a three-paragraph intelligence brief: the claim's contextual background, the arc of narrative evolution, and the analytical significance of the findings.

### Structured Report Output
The writer module renders the full **NarrativeReport** as seven markdown sections:
1. **Temporal Evolution** — time buckets with article counts, sentiment, frame, and voice data
2. **Perspective Landscape & Inflections** — all NLI clusters with corroboration levels
3. **Sentiment Timeline** — per-bucket scores and deltas with trend annotations
4. **Inflection Point Cards** — detailed cards per inflection with GDELT event correlation
5. **Frame Evolution Log** — frame labels per bucket with change flags
6. **Voice Composition Shifts** — dominant cluster per bucket with shift flags
7. **Narrative Intelligence Summary** — the three-paragraph LLM-generated brief

An optional JSON export of the full structured report is also available.

---

## 9. LLM Strategy and Model Selection

Three LLMs are used with distinct roles, each assigned to a separate API rate-limit bucket to prevent contention:

| Role | Model | Rationale |
|------|-------|-----------|
| Orchestration and final synthesis | Llama 3.3 70B (Groq) | High reasoning capacity for routing decisions, query planning, and final report generation |
| Classification tasks | Llama 4 Scout 17B (Groq) | Efficient for structured extraction: claim extraction, cluster labeling, frame classification, source typing |
| Synthesis and summarization | Llama 3.1 8B (Groq) | Fast, cost-effective for context summary, research brief, and other synthesis tasks |

All LLM outputs are schema-validated using Pydantic models, ensuring structured and reliable downstream consumption.

---

## 10. How It All Comes Together

The system's key contribution is the integration of **semantic retrieval, NLI-based perspective clustering, and multi-track temporal analysis** into a single coherent pipeline. Rather than treating fact-checking as a classification problem, the system treats it as a **narrative reconstruction problem**: given a claim, reconstruct the landscape of evidence, perspectives, and media behavior that surrounds it.

The output is not a verdict alone — it is an auditable, time-resolved account of how the claim was received, contested, corroborated, and framed across sources and time, grounded in real-world events and structured for human review.

---

## 11. Verification

The end-to-end pipeline can be verified by:
- Submitting a known contested claim (e.g., a public health or political claim with documented counter-narratives) and confirming that:
  - Multiple perspective clusters are generated with distinct labels
  - At least one inflection point is detected and correlated with a real GDELT event
  - The ground truth tier matches expected classification (contested)
  - All seven report sections are populated with non-trivial content
- Checking that coverage gaps trigger additional retrieval loops before the max loop count is reached
- Verifying that primary source verdict differs from secondary-source-derived cluster conclusions when primary sources are available
