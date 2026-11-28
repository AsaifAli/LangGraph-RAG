from retrieval.rag_pipeline import _build_filters


def test_filters_include_tenant_and_document_scope():
    filters = _build_filters(["doc-1"], "poc_demo", "langgraph_rag")
    assert {"term": {"tenant_schema": "poc_demo_langgraph_rag"}} in filters
    assert {"term": {"platform_tenant_id": "langgraph_rag"}} in filters
    assert {"terms": {"document_id": ["doc-1"]}} in filters


def test_empty_document_scope_fails_closed():
    filters = _build_filters([], "poc_demo", "langgraph_rag")
    assert {"match_none": {}} in filters



def test_retrieval_filters_require_the_exact_owner_scope():
    owner_a = "a" * 64
    owner_b = "b" * 64
    filters_a = _build_filters(["doc-1"], "poc_demo", "langgraph_rag", owner_a)
    filters_b = _build_filters(["doc-1"], "poc_demo", "langgraph_rag", owner_b)

    assert {"term": {"owner_scope": owner_a}} in filters_a
    assert {"term": {"owner_scope": owner_b}} in filters_b
    assert filters_a != filters_b



def test_empty_document_scope_in_legacy_builder_fails_closed():
    from types import SimpleNamespace
    from retrieval.kb_filter_builder import build_kb_filter_expr

    request = SimpleNamespace(
        tenant_schema="poc_demo",
        platform_tenant_id="langgraph_rag",
        document_ids=[],
    )

    filters = build_kb_filter_expr(request)

    assert {"match_none": {}} in filters
