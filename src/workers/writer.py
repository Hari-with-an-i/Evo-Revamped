"""
Writer worker — renders NarrativeReport into structured final output.

If a NarrativeReport is present in state (produced by analyst_node), this worker
renders it as a JSON block followed by 7 labelled markdown sections.

If no report is available (analyst skipped or failed), falls back to the legacy
BaseWorker prose writer for graceful degradation.
"""
from __future__ import annotations

import json

from src.config import config
from src.schemas.narrative_report import NarrativeReport
from src.state import AgentState
from src.logger import get_logger

log = get_logger(__name__)


def writer_node(state: AgentState) -> dict:
    report_dict = state.get("narrative_report")

    if not report_dict:
        log.info("writer: no narrative_report found, using legacy writer")
        return _legacy_writer_node(state)

    try:
        report = NarrativeReport(**report_dict)
    except Exception:
        log.exception("writer: failed to deserialize NarrativeReport, using legacy writer")
        return _legacy_writer_node(state)

    rendered = _render_report(report)
    log.info("writer: rendered NarrativeReport (%d chars)", len(rendered))

    snap = report.claim_snapshot
    summary = (
        f"[writer] Report rendered: tier={snap.get('ground_truth_tier','?').upper()}, "
        f"sources={snap.get('source_count', 0)}, "
        f"inflections={len(report.inflection_point_cards)}, "
        f"clusters={len(report.perspective_landscape)}"
    )
    return {
        "final_output": rendered,
        "worker_outputs": [summary],
    }


def _render_report(report: NarrativeReport) -> str:
    """Render NarrativeReport as JSON block + 3 condensed markdown sections."""
    sections: list[str] = []

    # JSON block — opt-in for downstream API consumers (set REPORT_INCLUDE_JSON=true)
    if config.REPORT_INCLUDE_JSON:
        sections.append("```json")
        sections.append(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        sections.append("```")
        sections.append("")

    sp_map = {pt.bucket_id: pt for pt in report.sentiment_timeline}
    fp_map = {pt.bucket_id: pt for pt in report.frame_evolution_log}
    vp_map = {pt.bucket_id: pt for pt in report.voice_composition_shifts}

    # --- Section 1: Temporal Evolution (Buckets, Sentiment, Frames & Voice) ---
    sections.append("## Section 1 — Temporal Evolution (Buckets, Sentiment, Frames & Voice)")
    if not report.time_buckets:
        sections.append("No temporal data available.")
    else:
        for b in report.time_buckets:
            bid = b.bucket_id
            date_range = f"{b.start_dt.date()} to {b.end_dt.date()}"
            sp = sp_map.get(bid)
            fp = fp_map.get(bid)
            vp = vp_map.get(bid)
            
            line_parts = [f"**Bucket {bid}** ({date_range})"]
            if sp:
                sign = "+" if sp.mean_sentiment >= 0 else ""
                line_parts.append(f"Sentiment: {sign}{sp.mean_sentiment:.2f} (Δ{sp.delta:+.2f})")
            if fp:
                shift_mark = " ⚠️ [SHIFT]" if fp.frame_changed else ""
                line_parts.append(f"Frame: {fp.frame_type}{shift_mark}")
            if vp:
                shift_mark = " ⚠️ [SHIFT]" if vp.voice_changed else ""
                line_parts.append(f"Voice: {vp.dominant_cluster_label}{shift_mark}")
            
            sections.append(" | ".join(line_parts))
    sections.append("")

    # --- Section 2: Perspective Landscape & Inflections ---
    sections.append("## Section 2 — Perspective Landscape & Inflections")
    if not report.perspective_landscape:
        sections.append("No perspective clusters detected.")
    else:
        sections.append("### Clusters")
        for p in report.perspective_landscape:
            corr_level = p.get("corroboration_level", "unverified")
            corr_count = p.get("corroboration_count", 0)
            label = p.get("label", "(unlabelled cluster)")
            claims = "; ".join(p.get("key_claims", [])[:2])
            sections.append(f"- **{label}**: {corr_level} ({corr_count} domains) — Key claims: {claims}")
    
    sections.append("\n### Inflection Points")
    if not report.inflection_point_cards:
        sections.append("No multi-track inflection points detected.")
    else:
        for ip in report.inflection_point_cards:
            sections.append(f"**Bucket {ip.bucket_id} ({ip.bucket_date_range})**")
            sections.append(f"- Dominant Perspective: {ip.dominant_cluster_label}")
            if ip.correlated_event:
                ev = ip.correlated_event
                sections.append(f"- Event: {ev.event_date} — {ev.description} [Plausibility: {ev.plausibility_score:.2f}]")
            sections.append(f"- Context: {ip.explanation}\n")

    # --- Section 3: Final Analysis ---
    snap = report.claim_snapshot
    tier = snap.get("ground_truth_tier", "unknown").upper()
    sections.append(f"## Section 3 — Final Analysis [{tier}]")
    sections.append(f"**Claim:** {snap.get('claim', '')}")
    sections.append(f"**Period:** {snap.get('time_period', 'unknown')}  |  **Sources:** {snap.get('source_count', 0)}\n")
    sections.append(report.narrative_intelligence_summary)

    return "\n".join(sections)


def _legacy_writer_node(state: AgentState) -> dict:
    """Fallback prose writer when analyst hasn't produced a NarrativeReport."""
    from src.workers.base_worker import BaseWorker

    class _FallbackWriter(BaseWorker):
        name = "writer"
        use_small_model = True
        system_prompt = (
            "You are an expert writer and editor. "
            "Using the context and instructions provided, produce clear, well-structured written content. "
            "Adapt your tone and format (prose, bullet points, report, email, etc.) to what is asked."
        )
        tools = []

    return _FallbackWriter().run(state)
