from __future__ import annotations

import asyncio

import retrieval.rag_pipeline as pipeline


def make_chunk(document_id: str, chunk_number: int, content: str, evidence_id: str = "E1"):
    return pipeline.RetrievedChunk(
        evidence_id=evidence_id,
        document_id=document_id,
        content=content,
        score=0.9,
        meta_data={
            "document_id": document_id,
            "chunk_id": f"{document_id}:{chunk_number}",
            "chunk_number": chunk_number,
            "source_name": f"{document_id}.pdf",
            "page_no": chunk_number,
        },
    )


def make_hit(document_id: str, chunk_number: int, content: str):
    source = {
        "document_id": document_id,
        "chunk_id": f"{document_id}:{chunk_number}",
        "chunk_number": chunk_number,
        "source_name": f"{document_id}.pdf",
        "page_no": chunk_number,
        "passage_text": content,
        "owner_scope": "a" * 64,
    }
    return {"_id": f"{document_id}:{chunk_number}", "_score": 0.2, "_source": source}


def test_context_window_query_is_limited_to_authorised_neighbouring_chunks():
    seed = make_chunk("doc-a", 5, "central evidence")

    body = pipeline._context_window_query_body(
        [seed],
        document_ids=["doc-a", "doc-b"],
        tenant_schema="schema",
        platform_tenant_id="tenant",
        owner_scope="a" * 64,
        window=1,
        size=2,
    )

    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"document_id": ["doc-a", "doc-b"]}} in filters
    assert {"term": {"owner_scope": "a" * 64}} in filters
    neighbour_filter = next(x["bool"]["should"] for x in filters if "bool" in x and "should" in x["bool"])
    assert neighbour_filter == [
        {"bool": {"filter": [
            {"term": {"document_id": "doc-a"}},
            {"range": {"chunk_number": {"gte": 4, "lte": 6}}},
        ]}}
    ]


def test_context_window_expansion_appends_adjacent_chunks_without_replacing_ranked_hits(monkeypatch):
    seed = make_chunk("doc-a", 5, "central evidence")
    calls = []

    class FakeClient:
        def search(self, *, index, body):
            calls.append((index, body))
            return {"hits": {"hits": [
                make_hit("doc-a", 4, "preceding context"),
                make_hit("doc-a", 5, "central evidence"),
                make_hit("doc-a", 6, "following context"),
            ]}}

    monkeypatch.setattr(pipeline, "build_opensearch_client", lambda: FakeClient())
    monkeypatch.setattr(pipeline, "_to_thread_with_retry", lambda fn, *args, **kwargs: _run_sync(fn, *args, **kwargs))

    result = asyncio.run(
        pipeline._expand_context_chunks(
            [seed],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a"],
            owner_scope="a" * 64,
            window=1,
            max_extra=2,
        )
    )

    assert [chunk.content for chunk in result] == [
        "central evidence",
        "preceding context",
        "following context",
    ]
    assert result[1].meta_data["context_only"] is True
    assert result[1].meta_data["context_for"] == ["E1"]
    assert result[2].meta_data["context_only"] is True
    assert len(calls) == 1


def _run_sync(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def test_context_expansion_does_not_query_without_authorised_document_ids(monkeypatch):
    def unexpected_client():
        raise AssertionError("Context expansion must not query without a document allow-list.")

    monkeypatch.setattr(pipeline, "build_opensearch_client", unexpected_client)

    result = asyncio.run(
        pipeline._expand_context_chunks(
            [make_chunk("doc-a", 5, "central evidence")],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=[],
            owner_scope="a" * 64,
            window=1,
            max_extra=2,
        )
    )

    assert result == [make_chunk("doc-a", 5, "central evidence")]


def test_context_expansion_ignores_seed_chunks_outside_the_authorised_scope(monkeypatch):
    def unexpected_client():
        raise AssertionError("An out-of-scope seed must not trigger context retrieval.")

    monkeypatch.setattr(pipeline, "build_opensearch_client", unexpected_client)

    result = asyncio.run(
        pipeline._expand_context_chunks(
            [make_chunk("private-chat-b", 2, "not in this thread")],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["active-chat-a"],
            owner_scope="a" * 64,
            window=1,
            max_extra=2,
        )
    )

    assert result == []
