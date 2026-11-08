
"""Supervisor-led multi-agent orchestration for EvidenceFlow.

This module preserves the existing production LangGraph workflow and adds
an independent Evidence Auditor after synthesis.

Existing specialist responsibilities:
- Supervisor: existing intent router and delegation logic.
- Knowledge Research: sparse-first retrieval and self-correction.
- Web Research: Tavily-backed external research.
- Primary Source: read-only MCP research.
- Chunk Analysts: parallel evidence extraction.
- Document Summarizers: document-level rollups.
- Synthesis: grounded final-answer generation.
- Evidence Auditor: independent post-synthesis review.

The same LLM backend may be shared across specialist roles. Role separation
comes from distinct responsibilities, prompts, graph nodes and outputs.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from agents.custom_langgraph_poc import (
    astream_langgraph_turn as _core_astream_langgraph_turn,
    build_langgraph_agent as _build_core_agent,
    run_langgraph_turn as _core_run_langgraph_turn,
)
from agents.specialist_agents import (
    EvidenceAuditDecision,
    build_evidence_context,
    run_evidence_auditor,
)
from retrieval.rag_pipeline import (
    DEFAULT_BACKEND,
    DEFAULT_TOP_K,
    PLATFORM_TENANT_ID,
    TENANT_SCHEMA,
    build_langchain_model,
)
from tools.primary_source_mcp import KatzillaMCPClient


AUDIT_ABSTENTION_MESSAGE = (
    "I couldn't safely return this answer because the independent evidence "
    "review could not verify one or more material claims against the "
    "available sources. Please narrow the question or provide additional "
    "source material."
)


def build_langgraph_agent(
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    document_ids_provider: Callable[[], list[str]] | None = None,
    document_names_provider: Callable[[], dict[str, str]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    backend: str = DEFAULT_BACKEND,
    checkpointer: Any | None = None,
    gateway_token: str = "",
    primary_source_mcp: KatzillaMCPClient | None = None,
    audit_enabled: bool = True,
) -> dict[str, Any]:
    """Build the existing EvidenceFlow graph with an independent auditor.

    The auditor model is initialized lazily on the first evidence-backed
    turn. This avoids an additional model initialization at application
    startup and reuses the configured backend and gateway token.
    """
    built = _build_core_agent(
        tenant_schema=tenant_schema,
        platform_tenant_id=platform_tenant_id,
        document_ids_provider=document_ids_provider,
        document_names_provider=document_names_provider,
        top_k=top_k,
        backend=backend,
        checkpointer=checkpointer,
        gateway_token=gateway_token,
        primary_source_mcp=primary_source_mcp,
    )

    built["evidence_audit_enabled"] = bool(audit_enabled)
    built["evidence_audit_backend"] = backend
    built["evidence_audit_gateway_token"] = gateway_token
    built["evidence_auditor_model"] = None

    return built


def _get_evidence_auditor_model(built: dict[str, Any]) -> Any:
    """Create and cache the auditor's model instance."""
    model = built.get("evidence_auditor_model")

    if model is None:
        model = build_langchain_model(
            built.get("evidence_audit_backend", DEFAULT_BACKEND),
            gateway_token=built.get("evidence_audit_gateway_token", ""),
        )
        built["evidence_auditor_model"] = model

    return model


def _add_agent_trace(
    outcome: dict[str, Any],
    *,
    audit_decision: str,
    audit_issues: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Attach an honest execution trace and UI-compatible delegations."""
    trace: list[dict[str, str]] = []
    delegations = list(outcome.get("delegations") or [])

    def add_role(
        name: str,
        responsibility: str,
        result: str,
    ) -> None:
        trace.append(
            {
                "agent": name,
                "responsibility": responsibility,
                "status": "completed",
            }
        )
        delegations.append(
            {
                "subagent_type": name.lower().replace(" ", "-"),
                "description": responsibility,
                "result": result,
            }
        )

    add_role(
        "Supervisor Agent",
        "Classify the request and select the appropriate research branches.",
        "Request routing completed.",
    )

    if outcome.get("used_knowledge_base"):
        chunk_count = len(outcome.get("chunks") or [])
        add_role(
            "Knowledge Research Agent",
            "Retrieve and analyze internal knowledge-base evidence.",
            (
                f"Knowledge-base research completed using {chunk_count} "
                "retrieved evidence chunks."
            ),
        )

    if outcome.get("used_web"):
        add_role(
            "Web Research Agent",
            "Gather and summarize external web evidence.",
            "External web research completed.",
        )

    if outcome.get("used_primary_source"):
        add_role(
            "Primary Source Agent",
            "Query official primary-source data through the MCP connector.",
            "Primary-source research completed.",
        )

    if (
        outcome.get("used_knowledge_base")
        or outcome.get("used_web")
        or outcome.get("used_primary_source")
    ):
        add_role(
            "Synthesis Agent",
            "Combine specialist findings into a grounded answer.",
            "Evidence synthesis completed.",
        )

    if audit_decision in {"PASS", "ABSTAIN", "ERROR"}:
        status = "completed" if audit_decision == "PASS" else "review_required"
        trace.append(
            {
                "agent": "Evidence Auditor",
                "responsibility": "Independently verify material claims against source evidence.",
                "status": status,
            }
        )
        delegations.append(
            {
                "subagent_type": "evidence-auditor",
                "description": "Independent post-synthesis evidence review",
                "result": (
                    "PASS: no material evidence problem detected."
                    if audit_decision == "PASS"
                    else "ABSTAIN: " + "; ".join(audit_issues)
                ),
            }
        )

    outcome["agent_trace"] = trace
    outcome["delegations"] = delegations
    return outcome


async def _apply_evidence_audit(
    outcome: dict[str, Any],
    built: dict[str, Any],
    *,
    question: str,
) -> dict[str, Any]:
    """Audit evidence-backed answers before returning them."""
    if not built.get("evidence_audit_enabled", True):
        outcome["evidence_audit"] = {"decision": "DISABLED", "issues": []}
        return _add_agent_trace(outcome, audit_decision="DISABLED")

    source_used = bool(
        outcome.get("used_knowledge_base")
        or outcome.get("used_web")
        or outcome.get("used_primary_source")
    )

    if not source_used:
        outcome["evidence_audit"] = {
            "decision": "SKIPPED",
            "issues": [],
            "reason": "No external or knowledge-base research branch was used.",
        }
        return _add_agent_trace(outcome, audit_decision="SKIPPED")

    # The existing deterministic verifier has already rejected insufficient
    # KB evidence. An additional model call cannot turn that refusal into PASS.
    if outcome.get("abstained"):
        outcome["evidence_audit"] = {
            "decision": "SKIPPED",
            "issues": [],
            "reason": "The core evidence workflow already abstained.",
        }
        return _add_agent_trace(outcome, audit_decision="SKIPPED")

    try:
        model = _get_evidence_auditor_model(built)
        evidence_context = build_evidence_context(outcome)

        decision: EvidenceAuditDecision = await run_evidence_auditor(
            model,
            question=question,
            draft_answer=str(outcome.get("final_answer") or ""),
            evidence_context=evidence_context,
        )

        audit_decision = decision.decision
        audit_issues = decision.issues

    except Exception as exc:
        # Do not expose provider exception details or silently skip the audit.
        audit_decision = "ABSTAIN"
        audit_issues = (
            f"Independent evidence review failed ({type(exc).__name__}).",
        )

    outcome["evidence_audit"] = {
        "decision": audit_decision,
        "issues": list(audit_issues),
    }

    if audit_decision != "PASS":
        outcome["final_answer"] = AUDIT_ABSTENTION_MESSAGE
        outcome["abstained"] = True
        outcome["abstention_reason"] = "EVIDENCE_AUDIT_ABSTAIN"

        quality = dict(outcome.get("quality") or {})
        quality["quality_label"] = "AUDIT_ABSTAINED"
        quality["evidence_audit"] = "ABSTAIN"
        outcome["quality"] = quality

        trust = dict(outcome.get("trust") or {})
        trust["label"] = "REVIEW"
        trust["evidence_audit"] = "ABSTAIN"
        trust["audit_issue_count"] = len(audit_issues)
        outcome["trust"] = trust

    else:
        quality = dict(outcome.get("quality") or {})
        quality["evidence_audit"] = "PASS"
        outcome["quality"] = quality

        trust = dict(outcome.get("trust") or {})
        trust["evidence_audit"] = "PASS"
        outcome["trust"] = trust

    return _add_agent_trace(
        outcome,
        audit_decision=audit_decision,
        audit_issues=audit_issues,
    )


async def run_langgraph_turn(
    built: dict[str, Any],
    query: str,
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Run the core workflow followed by the independent evidence auditor."""
    outcome = await _core_run_langgraph_turn(
        built,
        query,
        thread_id=thread_id,
    )

    return await _apply_evidence_audit(
        outcome,
        built,
        question=query,
    )


async def astream_langgraph_turn(
    built: dict[str, Any],
    query: str,
    *,
    thread_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Stream statuses, but withhold answer tokens until evidence review passes.

    This is deliberate: an unsupported draft must not appear in the UI before
    the independent auditor has had a chance to reject it.
    """
    buffered_tokens: list[str] = []

    async for event in _core_astream_langgraph_turn(
        built,
        query,
        thread_id=thread_id,
    ):
        event_type = event.get("type")

        if event_type == "token":
            token = event.get("text", "")
            if token:
                buffered_tokens.append(token)
            continue

        if event_type == "final":
            if built.get("evidence_audit_enabled", True):
                yield {
                    "type": "status",
                    "text": "🧪 Independent evidence auditor reviewing the answer",
                }

            audited = await _apply_evidence_audit(
                event,
                built,
                question=query,
            )

            # Only release the provisional synthesis tokens after PASS.
            if (
                audited.get("evidence_audit", {}).get("decision") == "PASS"
            ):
                for token in buffered_tokens:
                    yield {"type": "token", "text": token}

            audited["type"] = "final"
            yield audited
            return

        yield event
