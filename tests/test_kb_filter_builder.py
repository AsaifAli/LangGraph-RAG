from retrieval.rag_pipeline import _build_filters


def test_filters_include_tenant_and_document_scope():
    filters = _build_filters(["doc-1"], "poc_demo", "langgraph_rag")
    assert {"term": {"tenant_schema": "poc_demo_langgraph_rag"}} in filters
    assert {"term": {"platform_tenant_id": "langgraph_rag"}} in filters
    assert {"terms": {"document_id": ["doc-1"]}} in filters


def test_empty_document_scope_does_not_expand_scope():
    filters = _build_filters([], "poc_demo", "langgraph_rag")
    assert not any("document_id" in condition.get("terms", {}) for condition in filters)
