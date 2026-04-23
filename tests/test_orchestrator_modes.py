"""Tests for the orchestrator's three operating modes."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage

from src.orchestrator.orchestrator import OrchestratorDecision, orchestrator_node
from src.schemas import TargetedQuery


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
        "task": "analyse claim",
        "plan": [],
        "worker_outputs": [],
        "next": "",
        "instructions": "",
        "primary_evidence_exists": None,
        "primary_verdict": None,
        "narrative_report": None,
        "analysis_complete": None,
    }
    base.update(overrides)
    return base


def _mock_llm(decision: OrchestratorDecision) -> MagicMock:
    """Return a mock that _get_llm() will return, pre-configured with invoke response."""
    m = MagicMock()
    m.invoke.return_value = decision
    return m


# ---------------------------------------------------------------------------
# MODE A — input parsing
# ---------------------------------------------------------------------------

@patch("src.orchestrator.orchestrator._get_llm")
def test_mode_a_writes_claim(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(OrchestratorDecision(
        reasoning="Parsing input.",
        next="broad_context_fetch",
        parsed_claim="5G towers spread COVID-19.",
        parsed_entities="5G, COVID-19",
        parsed_timeframe="",
        claim_complexity="simple",
    ))

    result = orchestrator_node(_base_state())

    assert result["claim"] == "5G towers spread COVID-19."
    assert "5G" in result["entities"]
    assert result["claim_complexity"] == "simple"
    assert result["next"] == "broad_context_fetch"


@patch("src.orchestrator.orchestrator._get_llm")
def test_mode_a_does_not_overwrite_existing_claim(mock_get_llm):
    """If claim is already set, MODE A fields should not be written."""
    mock_get_llm.return_value = _mock_llm(OrchestratorDecision(
        reasoning="Planning queries.",
        next="tavily_targeted",
        new_targeted_queries=[
            TargetedQuery(query="5G COVID link", tool="tavily", dimension="causal mechanism")
        ],
    ))

    state = _base_state(
        claim="5G towers spread COVID-19.",
        context_summary={"who": "WHO", "what": "claim", "when": "2020",
                         "competing_narratives": "", "claim_dimensions": "causal"},
        retrieval_loop_count=0,
        retrieval_complete=False,
    )
    result = orchestrator_node(state)

    assert "claim" not in result


# ---------------------------------------------------------------------------
# MODE B — query planning
# ---------------------------------------------------------------------------

@patch("src.orchestrator.orchestrator._get_llm")
def test_mode_b_writes_targeted_queries(mock_get_llm):
    queries = [
        TargetedQuery(query="5G towers COVID-19", tool="tavily", dimension="causal mechanism"),
        TargetedQuery(query="5G COVID fact check", tool="gdelt_commoncrawl", dimension="fact check"),
        TargetedQuery(query="5G technology Wikipedia", tool="scholar_wiki", dimension="technical background"),
    ]
    mock_get_llm.return_value = _mock_llm(OrchestratorDecision(
        reasoning="Generating targeted queries.",
        next="tavily_targeted",
        new_targeted_queries=queries,
        retrieval_complete=False,
    ))

    state = _base_state(
        claim="5G towers spread COVID-19.",
        context_summary={"who": "WHO", "what": "5G claim", "when": "2020",
                         "competing_narratives": "", "claim_dimensions": "causal"},
        retrieval_loop_count=0,
        retrieval_complete=False,
    )
    result = orchestrator_node(state)

    assert len(result["targeted_queries"]) == 3
    assert result["targeted_queries"][0]["tool"] == "tavily"
    assert result["retrieval_loop_count"] == 1
    assert result["next"] == "tavily_targeted"


# ---------------------------------------------------------------------------
# MODE C — loop evaluation
# ---------------------------------------------------------------------------

@patch("src.orchestrator.orchestrator._get_llm")
def test_mode_c_sets_retrieval_complete_when_sufficient(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(OrchestratorDecision(
        reasoning="Enough articles — advancing.",
        next="researcher",
        retrieval_complete=True,
    ))

    articles = [{"id": str(i), "title": f"Article {i}", "body": "",
                 "source_name": "src", "source_url": f"http://ex.com/{i}",
                 "published_at": "2024-01-01T00:00:00+00:00",
                 "outlet_type": "unknown", "query_used": "q"} for i in range(12)]

    state = _base_state(
        claim="5G towers spread COVID-19.",
        context_summary={"who": "WHO", "what": "5G", "when": "2020",
                         "competing_narratives": "", "claim_dimensions": ""},
        retrieved_articles=articles,
        retrieval_loop_count=1,
        retrieval_complete=False,
    )
    result = orchestrator_node(state)

    assert result["retrieval_complete"] is True
    assert result["next"] == "researcher"


@patch("src.orchestrator.orchestrator._get_llm")
def test_mode_c_routes_to_finish_after_writer(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(OrchestratorDecision(
        reasoning="Writer done.",
        next="FINISH",
        retrieval_complete=True,
    ))

    state = _base_state(
        claim="5G towers spread COVID-19.",
        context_summary={"who": "WHO", "what": "5G", "when": "2020",
                         "competing_narratives": "", "claim_dimensions": ""},
        retrieval_complete=True,
        worker_outputs=["[researcher]\nDone.", "[writer]\nReport produced."],
    )
    result = orchestrator_node(state)

    assert result["next"] == "FINISH"
