
"""Tests for EvidenceFlow's specialist-agent utilities."""

import json

from agents.specialist_agents import (
    build_evidence_context,
    parse_audit_response,
)


def test_valid_pass_decision():
    response = json.dumps(
        {
            "decision": "PASS",
            "issues": [],
        }
    )

    result = parse_audit_response(response)

    assert result.decision == "PASS"
    assert result.issues == ()


def test_valid_abstain_decision():
    response = json.dumps(
        {
            "decision": "ABSTAIN",
            "issues": ["The stated amount is not supported by the source."],
        }
    )

    result = parse_audit_response(response)

    assert result.decision == "ABSTAIN"
    assert len(result.issues) == 1
    assert "not supported" in result.issues[0]


def test_markdown_fenced_json_is_supported():
    response = """```json
{
    "decision": "PASS",
    "issues": []
}
```"""

    result = parse_audit_response(response)

    assert result.decision == "PASS"


def test_malformed_json_fails_closed():
    result = parse_audit_response("This answer looks correct.")

    assert result.decision == "ABSTAIN"
    assert result.issues


def test_unsupported_decision_fails_closed():
    response = json.dumps(
        {
            "decision": "MAYBE",
            "issues": [],
        }
    )

    result = parse_audit_response(response)

    assert result.decision == "ABSTAIN"


def test_pass_with_issues_fails_closed():
    response = json.dumps(
        {
            "decision": "PASS",
            "issues": ["A material claim is unsupported."],
        }
    )

    result = parse_audit_response(response)

    assert result.decision == "ABSTAIN"
    assert result.issues


def test_missing_issues_field_fails_closed():
    response = json.dumps({"decision": "PASS"})

    result = parse_audit_response(response)

    assert result.decision == "ABSTAIN"


def test_evidence_context_includes_knowledge_chunks():
    outcome = {
        "chunks": [
            {
                "evidence_id": "E1",
                "document_id": "doc-123",
                "source_name": "policy.pdf",
                "content": "The policy limit is CAD 500,000.",
            }
        ]
    }

    context = build_evidence_context(outcome)

    assert "E1" in context
    assert "doc-123" in context
    assert "CAD 500,000" in context


def test_evidence_context_includes_web_and_primary_sources():
    outcome = {
        "chunks": [],
        "web_evidence": "External source: official report published in 2026.",
        "primary_source_evidence": "Official dataset reports a value of 42.",
    }

    context = build_evidence_context(outcome)

    assert "official report published in 2026" in context
    assert "value of 42" in context


def test_evidence_context_handles_retrieved_chunk_objects():
    class FakeChunk:
        evidence_id = "E7"
        document_id = "doc-456"
        source_name = "schedule.pdf"
        content = "The deductible is CAD 10,000."

    context = build_evidence_context({"chunks": [FakeChunk()]})

    assert "E7" in context
    assert "CAD 10,000" in context


def test_empty_evidence_is_explicitly_marked():
    context = build_evidence_context({"chunks": []})

    assert "NO VERIFIABLE SOURCE EVIDENCE" in context


def test_evidence_context_respects_character_limit():
    outcome = {
        "chunks": [
            {
                "evidence_id": "E1",
                "document_id": "doc-1",
                "content": "A" * 10000,
            },
            {
                "evidence_id": "E2",
                "document_id": "doc-2",
                "content": "B" * 10000,
            },
        ]
    }

    context = build_evidence_context(outcome, max_chars=1000)

    assert len(context) < 1500
    assert "truncated" in context.lower()
