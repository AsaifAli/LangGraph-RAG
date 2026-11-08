"""Integration smoke tests for EvidenceFlow's multi-agent wrapper."""

import json
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage

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