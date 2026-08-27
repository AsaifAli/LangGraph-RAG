from retrieval import rag_pipeline as rp


def test_sparse_first_defaults():
    assert rp.OPENSEARCH_NEURAL_SPARSE is True
    assert rp.RRF_K == 60
    assert rp.RERANK_ENABLED is True


def test_opensearch_index_name_is_safe():
    name = rp.opensearch_index_name("Portfolio Demo", "agentic/rag")
    assert " " not in name
    assert "/" not in name
    assert name.endswith("evidenceflow_documents")


def test_neural_sparse_query_uses_no_dense_vector_clause():
    body = rp._neural_query_body("approval criteria", ["doc-1"], "schema", "tenant", 5)
    raw = str(body).lower()
    assert "neural_sparse" in raw
    assert "knn" not in raw
    assert "dense" not in raw
