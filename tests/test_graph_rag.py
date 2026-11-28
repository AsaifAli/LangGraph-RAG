from __future__ import annotations

import asyncio

import retrieval.rag_pipeline as pipeline
from retrieval.graph_rag import extract_entity_terms


def _chunk(document_id: str, chunk_number: int, entities: list[str]):
    return pipeline.RetrievedChunk(
        evidence_id="E1",
        document_id=document_id,
        content="Seed passage",
        score=0.03,
        meta_data={
            "document_id": document_id,
            "chunk_id": f"{document_id}:{chunk_number}",
            "chunk_number": chunk_number,
            "entity_terms": entities,
        },
    )


def _hit(document_id: str, chunk_number: int, entities: list[str]):
    return {
        "_id": f"{document_id}:{chunk_number}",
        "_score": 1.0,
        "_source": {
            "document_id": document_id,
            "chunk_id": f"{document_id}:{chunk_number}",
            "chunk_number": chunk_number,
            "source_name": f"{document_id}.pdf",
            "passage_text": f"Related passage {chunk_number}",
            "entity_terms": entities,
        },
    }


def test_entity_extraction_keeps_named_entities_and_acronyms_without_sentence_fillers():
    terms = extract_entity_terms(
        "The Northbridge Holdings agreement references FDA, ISO 27001 and OpenAI. "
        "This section also mentions the Northbridge team."
    )
    assert "northbridge holdings" in terms
    assert "northbridge" in terms
    assert "fda" in terms
    assert "iso 27001" in terms
    assert "openai" in terms
    assert "the" not in terms
    assert "this" not in terms
    assert "agreement" not in terms
    assert "payment" not in terms
    assert "usd" not in terms


class _MappingClient:
    def __init__(self):
        self.mapping = None

    class _Indices:
        def __init__(self, outer):
            self.outer = outer

        def put_mapping(self, **kwargs):
            self.outer.mapping = kwargs

    @property
    def indices(self):
        return self._Indices(self)


def test_entity_terms_are_indexed_in_new_and_existing_opensearch_mappings():
    body = pipeline._new_index_mapping_body(sparse_enabled=False)
    assert body["mappings"]["properties"]["entity_terms"] == {"type": "keyword"}
    client = _MappingClient()
    pipeline._ensure_entity_mapping(client, "existing-index")
    assert client.mapping["index"] == "existing-index"
    assert client.mapping["body"]["properties"]["entity_terms"] == {"type": "keyword"}


def test_graph_query_body_reuses_the_authorized_document_and_owner_filters():
    body = pipeline._graph_query_body(
        ["northbridge", "fda"],
        document_ids=["global-doc", "active-thread-doc"],
        tenant_schema="schema",
        platform_tenant_id="tenant",
        owner_scope="a" * 64,
        exclude_chunk_ids=["active-thread-doc:1"],
        size=6,
    )
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"document_id": ["global-doc", "active-thread-doc"]}} in filters
    assert {"term": {"owner_scope": "a" * 64}} in filters
    assert {"term": {"tenant_schema": "schema_tenant"}} in filters
    assert body["query"]["bool"]["must"] == [
        {"terms": {"entity_terms": ["northbridge", "fda"]}}
    ]
    assert body["query"]["bool"]["must_not"] == [
        {"ids": {"values": ["active-thread-doc:1"]}}
    ]


def test_graph_expansion_follows_shared_entities_with_a_bounded_second_hop(monkeypatch):
    calls = []

    class FakeClient:
        def search(self, *, index, body):
            calls.append((index, body))
            if len(calls) == 1:
                return {"hits": {"hits": [
                    _hit("doc-b", 2, ["northbridge", "fda", "gdpr"]),
                    _hit("other-thread-doc", 1, ["northbridge"]),
                ]}}
            return {"hits": {"hits": [
                _hit("doc-c", 4, ["gdpr", "openai"]),
            ]}}

    monkeypatch.setattr(pipeline, "build_opensearch_client", lambda: FakeClient())
    result = asyncio.run(
        pipeline._expand_graph_candidates(
            [_chunk("doc-a", 1, ["northbridge"])],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a", "doc-b", "doc-c"],
            owner_scope="a" * 64,
            max_hops=2,
            max_entities=8,
            max_chunks=4,
        )
    )
    assert [item.document_id for item in result] == ["doc-a", "doc-b", "doc-c"]
    assert result[1].meta_data["graph_expanded"] is True
    assert result[1].meta_data["graph_hop"] == 1
    assert result[1].meta_data["graph_shared_entities"] == ["northbridge"]
    assert result[2].meta_data["graph_hop"] == 2
    assert len(calls) == 2
    for _, body in calls:
        filters = body["query"]["bool"]["filter"]
        assert {"terms": {"document_id": ["doc-a", "doc-b", "doc-c"]}} in filters
        assert {"term": {"owner_scope": "a" * 64}} in filters


def test_graph_expansion_never_queries_without_authorized_documents_or_seed_entities(monkeypatch):
    def unexpected_client():
        raise AssertionError("Graph retrieval must not query without an authorized graph seed.")

    monkeypatch.setattr(pipeline, "build_opensearch_client", unexpected_client)
    no_scope = asyncio.run(
        pipeline._expand_graph_candidates(
            [_chunk("doc-a", 1, ["northbridge"])],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=[],
            owner_scope="a" * 64,
        )
    )
    no_entities = asyncio.run(
        pipeline._expand_graph_candidates(
            [_chunk("doc-a", 1, [])],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a"],
            owner_scope="a" * 64,
        )
    )
    assert no_scope == []
    assert [item.document_id for item in no_entities] == ["doc-a"]


def test_graph_expansion_ignores_out_of_scope_hits(monkeypatch):
    class FakeClient:
        def search(self, *, index, body):
            return {"hits": {"hits": [_hit("private-chat-doc", 2, ["northbridge"])]}}

    monkeypatch.setattr(pipeline, "build_opensearch_client", lambda: FakeClient())
    result = asyncio.run(
        pipeline._expand_graph_candidates(
            [_chunk("active-chat-doc", 1, ["northbridge"])],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["active-chat-doc"],
            owner_scope="a" * 64,
            max_hops=1,
            max_chunks=4,
        )
    )
    assert [item.document_id for item in result] == ["active-chat-doc"]



def test_ingestion_metadata_persists_normalized_entity_terms():
    metadata = pipeline._flatten_metadata(
        document_id="doc-a",
        owner_scope="local-development",
        tenant_schema="schema",
        platform_tenant_id="tenant",
        source_name="agreement.pdf",
        chunk_number=1,
        content="Northbridge Holdings entered an agreement with Acme Corporation under FDA rules. Payment is due in USD 500.",
    )

    assert "northbridge holdings" in metadata["entity_terms"]
    assert "northbridge" in metadata["entity_terms"]
    assert "acme corporation" in metadata["entity_terms"]
    assert "acme" in metadata["entity_terms"]
    assert "fda" in metadata["entity_terms"]
    assert "payment" not in metadata["entity_terms"]
    assert "usd 500" not in metadata["entity_terms"]


def test_search_result_rehydrates_entity_terms_for_graph_expansion():
    chunk = pipeline._source_to_chunk(
        _hit("doc-a", 1, ["Northbridge Holdings", "FDA"])
    )

    assert chunk.meta_data["entity_terms"] == ["northbridge holdings", "fda"]


def test_disabled_graph_retrieval_is_a_no_op_without_creating_a_client(monkeypatch):
    def unexpected_client():
        raise AssertionError("Disabled graph retrieval must not create an OpenSearch client.")

    monkeypatch.setattr(pipeline, "build_opensearch_client", unexpected_client)
    result = asyncio.run(
        pipeline._expand_graph_candidates(
            [_chunk("doc-a", 1, ["northbridge"])],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a"],
            owner_scope="a" * 64,
            enabled=False,
        )
    )

    assert [item.document_id for item in result] == ["doc-a"]


def test_retrieve_and_rerank_passes_graph_neighbours_to_the_existing_reranker(monkeypatch):
    seed = [
        _hit("doc-a", number, ["northbridge"] if number == 1 else [])
        for number in range(1, 31)
    ]
    neighbours = [
        _hit("doc-b", number, ["northbridge", f"regulation-{number}"])
        for number in range(2, 8)
    ]
    rerank_inputs = []

    class FakeClient:
        def search(self, *, index, body):
            return {"hits": {"hits": neighbours}}

    def fake_rerank(query, chunks, *, top_n):
        rerank_inputs.append((query, chunks, top_n))
        return [next(chunk for chunk in chunks if chunk.meta_data.get("graph_expanded"))]

    monkeypatch.setattr(pipeline, "build_opensearch_client", lambda: FakeClient())
    monkeypatch.setattr(pipeline, "ensure_opensearch_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "_search_once",
        lambda *args, **kwargs: (seed, seed),
    )
    monkeypatch.setattr(pipeline, "_rerank", fake_rerank)
    monkeypatch.setattr(pipeline, "RERANK_ENABLED", True)
    monkeypatch.setattr(pipeline, "GRAPH_RAG_MAX_CHUNKS", 6)

    result = asyncio.run(
        pipeline.retrieve_and_rerank(
            "What is the Northbridge position?",
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a", "doc-b"],
            top_k=1,
            owner_scope="a" * 64,
        )
    )

    assert len(rerank_inputs) == 1
    assert len(rerank_inputs[0][1]) == 36
    assert any(chunk.meta_data.get("graph_expanded") for chunk in rerank_inputs[0][1])
    assert len(result) == 1
    assert result[0].document_id == "doc-b"



def test_graph_search_failure_preserves_the_authorized_sparse_seed(monkeypatch):
    async def fail_search(*args, **kwargs):
        raise RuntimeError("OpenSearch graph query unavailable")

    monkeypatch.setattr(pipeline, "_to_thread_with_retry", fail_search)

    seed = _chunk("doc-a", 1, ["northbridge"])
    result = asyncio.run(
        pipeline._expand_graph_candidates(
            [seed],
            tenant_schema="schema",
            platform_tenant_id="tenant",
            document_ids=["doc-a"],
            owner_scope="a" * 64,
            client=object(),
        )
    )

    assert result == [seed]
    assert result[0].document_id == "doc-a"
