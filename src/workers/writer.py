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
    """Render NarrativeReport as JSON block + 7 markdown sections."""
    sections: list[str] = []

    # JSON block — opt-in for downstream API consumers (set REPORT_INCLUDE_JSON=true)
    if config.REPORT_INCLUDE_JSON:
        sections.append("```json")
        sections.append(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        sections.append("```")
        sections.append("")

    # --- Section 1: Claim Snapshot ---
    snap = report.claim_snapshot
    tier = snap.get("ground_truth_tier", "unknown").upper()
    sections.append(f"## Section 1 — Claim Snapshot [{tier}]")
    sections.append(f"**Claim:** {snap.get('claim', '')}")
    sections.append(f"**Period:** {snap.get('time_period', 'unknown')}  |  **Sources:** {snap.get('source_count', 0)}")
    sections.append("")

    # --- Section 2: Perspective Landscape ---
    sections.append("## Section 2 — Perspective Landscape")
    if not report.perspective_landscape:
        sections.append("No perspective clusters detected.")
    for p in report.perspective_landscape:
        corr_level = p.get("corroboration_level", "unverified")
        corr_count = p.get("corroboration_count", 0)
        sections.append(f"### {p.get('label', '(unlabelled cluster)')}")
        sections.append(f"Corroboration: **{corr_level}** ({corr_count} independent domain(s))")
        key_claims = p.get("key_claims", [])
        for claim in key_claims[:3]:
            sections.append(f"- {claim}")
        sections.append("")

    # --- Section 3: Sentiment Timeline ---
    sections.append("## Section 3 — Sentiment Timeline")
    for pt in report.sentiment_timeline:
        bar_len = max(0, round(abs(pt.mean_sentiment) * 10))
        bar = "█" * bar_len
        sign = "+" if pt.mean_sentiment >= 0 else "-"
        sections.append(
            f"Bucket {pt.bucket_id}: {sign}{bar} "
            f"(mean={pt.mean_sentiment:+.2f}, Δ={pt.delta:+.2f}) "
            f"[{pt.article_count} articles | entity: {pt.dominant_entity}]"
        )
    sections.append("")

    # --- Section 4: Inflection Point Cards ---
    sections.append("## Section 4 — Inflection Points")
    if not report.inflection_point_cards:
        sections.append("No multi-track inflection points detected.")
    for ip in report.inflection_point_cards:
        sections.append(f"### Inflection — Bucket {ip.bucket_id} ({ip.bucket_date_range})")
        sections.append(f"**Tracks signaling:** {', '.join(ip.tracks_signaling)}")
        sections.append(f"**Dominant perspective:** {ip.dominant_cluster_label}")
        if ip.correlated_event:
            ev = ip.correlated_event
            sections.append(
                f"**Correlated event ({ev.event_date}):** {ev.description} "
                f"[plausibility: {ev.plausibility_score:.2f}]"
            )
        sections.append(f"**Explanation:** {ip.explanation}")
        sections.append("")

    # --- Section 5: Frame Evolution Log ---
    sections.append("## Section 5 — Frame Evolution Log")
    for fp in report.frame_evolution_log:
        shift_flag = "  **[FRAME SHIFT]**" if fp.frame_changed else ""
        top_terms_str = ", ".join(fp.top_terms[:10])
        sections.append(
            f"Bucket {fp.bucket_id}: **{fp.frame_type}**{shift_flag}"
            + (f" — {top_terms_str}" if top_terms_str else "")
        )
    sections.append("")

    # --- Section 6: Voice Composition Shifts ---
    sections.append("## Section 6 — Voice Composition Shifts")
    for vp in report.voice_composition_shifts:
        shift_flag = "  **[VOICE SHIFT]**" if vp.voice_changed else ""
        comp_str = "  |  ".join(f"{k}: {v}" for k, v in vp.composition.items()) if vp.composition else "no data"
        sections.append(
            f"Bucket {vp.bucket_id}: dominant=**{vp.dominant_cluster_label}**{shift_flag}  [{comp_str}]"
        )
    sections.append("")

    # --- Section 7: Narrative Intelligence Summary ---
    sections.append("## Section 7 — Narrative Intelligence Summary")
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
