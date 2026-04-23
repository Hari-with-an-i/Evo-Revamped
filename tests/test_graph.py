"""Smoke tests for the orchestrator-worker graph."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage

from src.schemas import TargetedQuery, ContextSummary
from src.orchestrator.orchestrator import OrchestratorDecision
from src.graphs import build_graph


def _base_state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="test")],
        "raw_input": "5G towers spread COVID-19.",
        "input_type": "claim",
        "claim": "",
        "entities": [],
        "timeframe": None,
        "claim_complexity": None,
        "context_summary": None,
        "retrieved_articles": [],
        "targeted_queries": [],
        "retrieval_loop_count": 0,
        "retrieval_complete": False,
        "perspective_clusters": [],
        "coverage_gaps": [],
        "evaluation_ready": None,
        "anchor_articles": [],
        "narrative_report": None,
        "analysis_complete": None,
        "task": "analyse claim",
        "plan": [],
        "worker_outputs": [],
        "next": "",
        "instructions": "",
        "primary_evidence_exists": None,
        "primary_verdict": None,
    }
    base.update(overrides)
    return base


def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


@patch("src.workers.context_builder._get_llm")
@patch("src.workers.broad_context_fetch.tavily_search")
@patch("src.orchestrator.orchestrator._get_llm")
def test_full_pipeline_reaches_finish(mock_orch_get_llm, mock_tavily, mock_cb_get_llm):
    mock_tavily.invoke.return_value = []

    mock_cb_llm = MagicMock()
    mock_cb_llm.invoke.return_value = ContextSummary(
        who="WHO", what="5G claim", when="2020",
        competing_narratives="", claim_dimensions="causal",
    )
    mock_cb_get_llm.return_value = mock_cb_llm

    responses = [
        OrchestratorDecision(
            reasoning="Parsing.",
            next="broad_context_fetch",
            parsed_claim="5G towers spread COVID-19.",
            parsed_entities="5G, COVID-19",
            claim_complexity="simple",
        ),
        OrchestratorDecision(
            reasoning="Skipping Phase 2.",
            next="researcher",
            retrieval_complete=True,
        ),
        OrchestratorDecision(reasoning="Writing.", next="writer"),
        OrchestratorDecision(reasoning="Done.", next="FINISH"),
    ]
    mock_orch_llm = MagicMock()
    mock_orch_llm.invoke.side_effect = responses
    mock_orch_get_llm.return_value = mock_orch_llm

    with patch("src.workers.researcher.ResearcherWorker.run",
               return_value={"worker_outputs": ["[researcher]\nDone."]}), \
         patch("src.workers.writer._legacy_writer_node",
               return_value={"worker_outputs": ["[writer]\nReport."]}):

        graph = build_graph()
        result = graph.invoke(_base_state())

    assert result["next"] == "FINISH"
    assert result["claim"] == "5G towers spread COVID-19."
