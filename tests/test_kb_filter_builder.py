"""Unit tests for retrieval.kb_filter_builder.build_kb_filter_expr. Pure
function: builds a qdrant_client.models.Filter, no I/O."""

from uuid import UUID

from qdrant_client import models

from retrieval.entity_filters import entity_coexistence_conditions
from retrieval.kb_filter_builder import build_kb_filter_expr
from shared.constants import VectorDBConstants
from shared.entities import SanitizedQuery, SecureAgentRequest

DOC_ID = UUID("11111111-1111-1111-1111-111111111111")


def _request(**overrides) -> SecureAgentRequest:
    defaults = dict(
        query=SanitizedQuery(content="what is the sublimit?"),
        correlation_id="corr-1",
        platform_tenant_id="langgraph_rag",
        tenant_schema="poc_demo",
        document_ids=[DOC_ID],
    )
    defaults.update(overrides)
    return SecureAgentRequest(**defaults)


def test_filter_includes_tenant_and_document_conditions():
    req = _request()
    filt = build_kb_filter_expr(req)

    assert isinstance(filt, models.Filter)
    conditions = filt.must
    tenant_cond = next(c for c in conditions if c.key == VectorDBConstants.META_TENANT_SCHEMA_KEY)
    assert tenant_cond.match.value == "poc_demo_langgraph_rag"

    doc_cond = next(c for c in conditions if c.key == VectorDBConstants.META_DOCUMENT_ID_KEY)
    assert doc_cond.match.any == [str(DOC_ID)]


def test_no_document_ids_produces_empty_match_any():
    req = _request(document_ids=None)
    filt = build_kb_filter_expr(req)
    doc_cond = next(c for c in filt.must if c.key == VectorDBConstants.META_DOCUMENT_ID_KEY)
    assert doc_cond.match.any == []


def test_no_thread_scoping_condition_without_a_chat_scope():
    req = _request()
    filt = build_kb_filter_expr(req, scope=None)
    keys = [c.key for c in filt.must]
    assert VectorDBConstants.META_THREAD_ID_KEY not in keys


def test_person_and_org_ids_add_coexistence_conditions():
    req = _request(person_ids=["p1"], org_ids=["o1"])
    filt = build_kb_filter_expr(req)
    expected_extra = entity_coexistence_conditions(["p1"], ["o1"])
    # The tenant + document FieldConditions come first, then one coexistence
    # Filter per non-empty id list.
    assert len(filt.must) == 2 + len(expected_extra)


def test_tenant_scope_changes_with_tenant_identity():
    a = _request(platform_tenant_id="tenant-a", tenant_schema="schema")
    b = _request(platform_tenant_id="tenant-b", tenant_schema="schema")
    fa = build_kb_filter_expr(a)
    fb = build_kb_filter_expr(b)
    tenant_key = VectorDBConstants.META_TENANT_SCHEMA_KEY
    va = next(c.match.value for c in fa.must if c.key == tenant_key)
    vb = next(c.match.value for c in fb.must if c.key == tenant_key)
    assert va == "schema_tenant-a"
    assert vb == "schema_tenant-b"
    assert va != vb


def test_empty_document_scope_is_fail_closed():
    req = _request(document_ids=[])
    filt = build_kb_filter_expr(req)
    doc_cond = next(c for c in filt.must if c.key == VectorDBConstants.META_DOCUMENT_ID_KEY)
    assert doc_cond.match.any == []
