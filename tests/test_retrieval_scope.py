from __future__ import annotations

import asyncio

import retrieval.rag_pipeline as pipeline


def test_empty_scope_never_constructs_a_broad_search_query():
    lexical = pipeline._lexical_query_body(
        "what is the policy limit", [], "schema", "tenant", 10, "a" * 64
    )
    neural = pipeline._neural_query_body(
        "what is the policy limit", [], "schema", "tenant", 10, "a" * 64
    )

    assert {"match_none": {}} in lexical["query"]["bool"]["filter"]
    assert {"match_none": {}} in neural["query"]["bool"]["filter"]


def test_retrieve_and_rerank_returns_empty_without_opening_opensearch(monkeypatch):
    def unexpected_client():
        raise AssertionError("An empty authorised scope must not query OpenSearch.")

    monkeypatch.setattr(pipeline, "build_opensearch_client", unexpected_client)

    result = asyncio.run(
        pipeline.retrieve_and_rerank(
            "show all policy limits",
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=[],
            owner_scope="a" * 64,
        )
    )

    assert result == []


def test_self_correction_does_not_spend_llm_calls_when_scope_is_empty(monkeypatch):
    class UnexpectedPlanner:
        async def ainvoke(self, prompt):
            raise AssertionError("No query planning is allowed without authorised documents.")

    async def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("Empty-scope retrieval must short-circuit before search.")

    monkeypatch.setattr(pipeline, "retrieve_and_rerank", unexpected_retrieval)

    result, effective_query = asyncio.run(
        pipeline.retrieve_with_self_correction(
            "what does the document say?",
            model=UnexpectedPlanner(),
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=[],
            owner_scope="a" * 64,
        )
    )

    assert result == []
    assert effective_query == "what does the document say?"
