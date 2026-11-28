"""Integration smoke tests for EvidenceFlow's multi-agent wrapper."""

import json
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from agents import custom_langgraph_poc as core
from agents import multi_agent_orchestrator as orchestrator


class FakeModel:
    """Model stub for graph construction without provider credentials."""

    def bind_tools(self, tools):
        self.tools = tools
        return self


class FakeAuditModel:

    def __init__(
        self,
        decision="PASS",
        issues=None,
        error=None,
    ):
        self.decision = decision
        self.issues = issues or []
        self.error = error

    async def ainvoke(self, messages):

        if self.error:
            raise self.error

        return AIMessage(
            content=json.dumps(
                {
                    "decision": self.decision,
                    "issues": self.issues,
                }
            )
        )


@pytest.fixture
def kb_outcome():

    return {
        "final_answer": (
            "The policy limit is CAD 500,000 [EVID: E1]."
        ),
        "chunks": [
            {
                "evidence_id": "E1",
                "document_id": "doc-123",
                "source_name": "policy.pdf",
                "content": "The policy limit is CAD 500,000.",
            }
        ],
        "used_knowledge_base": True,
        "used_web": False,
        "used_primary_source": False,
        "abstained": False,
        "delegations": [],
        "quality": {},
        "trust": {},
    }


def test_build_agent_compiles_without_provider_calls(monkeypatch):

    fake_model = FakeModel()

    monkeypatch.setattr(
        core,
        "build_langchain_model",
        lambda *args, **kwargs: fake_model,
    )

    monkeypatch.setattr(
        core,
        "load_runtime_config",
        lambda: SimpleNamespace(
            web_search=SimpleNamespace(
                web_search_api_key=""
            )
        ),
    )

    built = orchestrator.build_langgraph_agent(
        document_ids_provider=lambda: ["doc-123"],
        document_names_provider=lambda: {
            "doc-123": "policy.pdf"
        },
        primary_source_mcp=SimpleNamespace(
            available=False
        ),
    )

    assert built["agent"] is not None

    assert callable(
        built["agent"].ainvoke
    )

    assert built["evidence_audit_enabled"] is True

    assert fake_model.tools


@pytest.mark.asyncio
async def test_auditor_pass_preserves_answer(
    monkeypatch,
    kb_outcome,
):

    monkeypatch.setattr(
        orchestrator,
        "_get_evidence_auditor_model",
        lambda built: FakeAuditModel("PASS"),
    )

    built = {
        "evidence_audit_enabled": True
    }

    original_answer = kb_outcome["final_answer"]

    result = await orchestrator._apply_evidence_audit(
        kb_outcome,
        built,
        question="What is the policy limit?",
    )

    assert result["evidence_audit"]["decision"] == "PASS"

    assert result["final_answer"] == original_answer

    assert result["abstained"] is False

    agents = [
        item["agent"]
        for item in result["agent_trace"]
    ]

    assert "Supervisor Agent" in agents

    assert "Knowledge Research Agent" in agents

    assert "Synthesis Agent" in agents

    assert "Evidence Auditor" in agents


@pytest.mark.asyncio
async def test_auditor_abstain_blocks_answer(
    monkeypatch,
    kb_outcome,
):

    monkeypatch.setattr(
        orchestrator,
        "_get_evidence_auditor_model",
        lambda built: FakeAuditModel(
            "ABSTAIN",
            [
                "The numerical claim is not supported."
            ],
        ),
    )

    built = {
        "evidence_audit_enabled": True
    }

    result = await orchestrator._apply_evidence_audit(
        kb_outcome,
        built,
        question="What is the policy limit?",
    )

    assert result["evidence_audit"]["decision"] == "ABSTAIN"

    assert result["abstained"] is True

    assert (
        result["abstention_reason"]
        == "EVIDENCE_AUDIT_ABSTAIN"
    )

    assert "couldn't safely return" in (
        result["final_answer"].lower()
    )

    assert (
        result["quality"]["quality_label"]
        == "AUDIT_ABSTAINED"
    )


@pytest.mark.asyncio
async def test_auditor_provider_failure_fails_closed(
    monkeypatch,
    kb_outcome,
):

    monkeypatch.setattr(
        orchestrator,
        "_get_evidence_auditor_model",
        lambda built: FakeAuditModel(
            error=RuntimeError(
                "simulated provider failure"
            )
        ),
    )

    built = {
        "evidence_audit_enabled": True
    }

    result = await orchestrator._apply_evidence_audit(
        kb_outcome,
        built,
        question="What is the policy limit?",
    )

    assert result["evidence_audit"]["decision"] == "ABSTAIN"

    assert result["abstained"] is True

    assert "RuntimeError" in (
        result["evidence_audit"]["issues"][0]
    )

    assert (
        "simulated provider failure"
        not in result["final_answer"]
    )


@pytest.mark.asyncio
async def test_direct_response_skips_evidence_audit(
    kb_outcome,
):

    outcome = {
        **kb_outcome,
        "used_knowledge_base": False,
        "used_web": False,
        "used_primary_source": False,
        "chunks": [],
    }

    result = await orchestrator._apply_evidence_audit(
        outcome,
        {
            "evidence_audit_enabled": True
        },
        question="Hello!",
    )

    assert result["evidence_audit"]["decision"] == "SKIPPED"

    assert result["abstained"] is False


@pytest.mark.asyncio
async def test_core_stream_executes_graph_once_and_emits_one_final(
    monkeypatch,
):
    """A fully consumed stream must not execute LangGraph twice."""

    class FakeStreamAgent:

        def __init__(self):
            self.calls = 0

        async def astream_events(
            self,
            *args,
            **kwargs,
        ):

            self.calls += 1

            yield {
                "event": "on_chain_start",
                "name": "LangGraph",
                "run_id": "root-run",
                "parent_ids": [],
                "metadata": {},
            }

            yield {
                "event": "on_chain_start",
                "name": "retrieve_kb",
                "run_id": "retrieve-run",
                "parent_ids": [
                    "root-run"
                ],
                "metadata": {},
            }

            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "run_id": "root-run",
                "parent_ids": [],
                "data": {
                    "output": {
                        "state": "complete"
                    }
                },
                "metadata": {},
            }

    agent = FakeStreamAgent()

    monkeypatch.setattr(
        core,
        "validate_query",
        lambda query: None,
    )

    monkeypatch.setattr(
        core,
        "make_token_tracker",
        lambda: object(),
    )

    monkeypatch.setattr(
        core,
        "_turn_input",
        lambda built, query: {
            "query": query
        },
    )

    monkeypatch.setattr(
        core,
        "_finalize_outcome",
        lambda state, tracker, web_tool: {
            "final_answer": "verified"
        },
    )

    events = [
        event
        async for event in core.astream_langgraph_turn(
            {
                "agent": agent,
                "web_search_tool": None,
            },
            "test question",
            thread_id="stream-regression",
        )
    ]

    assert agent.calls == 1

    assert sum(
        event.get("type") == "final"
        for event in events
    ) == 1

    assert events[-1]["final_answer"] == "verified"

class FixedAnswerModel:
    def __init__(self):
        self.tool_names = []
        self.calls = 0

    def bind_tools(self, tools):
        self.tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="The current public standard is described by the cited source [WEB: W1].")


class SuccessfulWebSearch:
    available = True

    async def search_structured(self, query):
        return SimpleNamespace(
            ok=True,
            text="A search provider summary.",
            sources=(
                {
                    "evidence_id": "W1",
                    "title": "Official standards page",
                    "url": "https://example.com/standards",
                    "content": "The standard was updated recently.",
                    "score": 0.95,
                },
            ),
            error_code=None,
        )


@pytest.mark.asyncio
async def test_explicit_web_request_bypasses_unrelated_document_retrieval():
    model = FixedAnswerModel()
    graph = core._build_graph(
        model=model,
        web_search_tool=SuccessfulWebSearch(),
        primary_source_mcp=SimpleNamespace(available=False),
    ).compile()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Search the web for the latest safety standards.")],
            "document_ids": ["unrelated-supplier-quote"],
            "document_names": {"unrelated-supplier-quote": "supplier-quote.pdf"},
        }
    )

    assert result["source_mode"] == "web"
    assert result["needs_kb"] is False
    assert result["web_search_status"] == "succeeded"
    assert result["web_sources"][0]["evidence_id"] == "W1"
    # The one model call is synthesis. The route tool-selection call is bypassed.
    assert model.calls == 1


@pytest.mark.asyncio
async def test_explicit_web_request_without_search_does_not_fall_back_to_documents():
    model = FixedAnswerModel()
    graph = core._build_graph(
        model=model,
        web_search_tool=None,
        primary_source_mcp=SimpleNamespace(available=False),
    ).compile()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Search the web for the latest safety standards.")],
            "document_ids": ["unrelated-supplier-quote"],
        }
    )

    assert result["source_mode"] == "web"
    assert result.get("needs_kb") is not True
    assert result["web_search_status"] == "unavailable"
    assert "did not substitute uploaded documents" in result["final_answer"]


def test_failed_web_search_is_not_reported_as_successful_external_research():
    result = core._finalize_outcome(
        {
            "final_answer": "External research could not be completed.",
            "source_mode": "web",
            "web_search_status": "failed",
            "web_search_error_code": "provider_error",
            "web_summary": "Web search is temporarily unavailable.",
            "web_sources": [],
            "needs_kb": False,
        },
        SimpleNamespace(usage_metadata={}),
        None,
    )

    assert result["used_web"] is False
    assert result["used_web_sources"] is False
    assert result["web_search_status"] == "failed"
    assert result["web_search_error_code"] == "provider_error"


@pytest.mark.asyncio
async def test_quote_market_comparison_requires_both_uploaded_evidence_and_web_results(monkeypatch):
    async def no_document_chunks(*args, **kwargs):
        return []

    monkeypatch.setattr(core, "fetch_all_document_chunks", no_document_chunks)
    model = FixedAnswerModel()
    graph = core._build_graph(
        model=model,
        web_search_tool=SuccessfulWebSearch(),
        primary_source_mcp=SimpleNamespace(available=False),
    ).compile()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Compare this supplier quotation with the going market rate and tell me if it is fair.")],
            "document_ids": ["supplier-quote"],
            "tenant_schema": "test_tenant",
            "platform_tenant_id": "test_platform",
            "owner_scope": "test_owner",
        }
    )

    assert result["source_mode"] == "both"
    assert result["needs_kb"] is True
    assert result["needs_web"] is True


def test_web_capability_is_reported_as_non_secret_configuration_state(monkeypatch):
    from types import SimpleNamespace
    from retrieval import rag_pipeline

    monkeypatch.setattr(
        rag_pipeline,
        "load_runtime_config",
        lambda: SimpleNamespace(web_search=SimpleNamespace(web_search_api_key="")),
    )
    capability = rag_pipeline.get_web_search_capability()

    assert capability == {"provider": "Tavily", "status": "not_configured", "available": False}
    assert "api_key" not in capability
