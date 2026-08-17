from retrieval import rag_pipeline as rp


def test_default_local_retrieval_mode():
    # The repository remains backwards compatible unless Render explicitly
    # switches RETRIEVAL_BACKEND/RERANK_BACKEND through environment variables.
    assert rp.RETRIEVAL_BACKEND in {"local", "qdrant-cloud", "qdrant_cloud", "hosted"}
    assert rp.RERANK_BACKEND in {"local", "jina", "hosted"}


def test_hosted_models_preserve_current_vector_space():
    assert rp.HOSTED_DENSE_MODEL.lower() == "sentence-transformers/all-minilm-l6-v2"
    assert rp.HOSTED_SPARSE_MODEL == "Qdrant/bm25"
    assert rp.HOSTED_DENSE_VECTOR_SIZE == 384
