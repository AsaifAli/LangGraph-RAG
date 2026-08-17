# Render Hosted Retrieval Profile

This deployment preserves the real retrieval architecture while moving only the heavyweight model inference out of the Render container.

## Runtime architecture

```text
Render Streamlit / LangGraph
        |
        +--> Qdrant Cloud Inference
        |       |- sentence-transformers/all-MiniLM-L6-v2 (dense, 384d)
        |       `- Qdrant/bm25 (sparse)
        |
        +--> Qdrant Query API -> Reciprocal Rank Fusion (RRF)
        |
        +--> Jina reranker API -> jina-reranker-v1-turbo-en
        |
        `--> shared LLM gateway / BYOK model
```

The dense model is intentionally the same `all-MiniLM-L6-v2` used by the local profile, so the vector space remains 384-dimensional. Qdrant Cloud Inference can host this model and its BM25 sparse model directly, and Qdrant performs dense+sparse hybrid fusion with RRF server-side.

The Render image therefore excludes the heavy local `torch`, `sentence-transformers`, and `fastembed` runtime path.

## Environment variables

```
RETRIEVAL_BACKEND=qdrant-cloud
RERANK_BACKEND=jina
HOSTED_DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HOSTED_SPARSE_MODEL=Qdrant/bm25
HOSTED_DENSE_VECTOR_SIZE=384
QDRANT_URL=<free Qdrant Cloud cluster>
QDRANT_API_KEY=<Qdrant API key>
JINA_API_KEY=<Jina API key>
```

The local Streamlit/Compose profile remains unchanged unless these backend variables are changed.

## First deployment

1. Create/verify a Qdrant Cloud Free Cluster with Cloud Inference enabled.
2. Create/seed the hybrid collection from this application.
3. Seed the synthetic demo documents: `python -m scripts.seed_demo_kb`.
4. Deploy this repository to Render using `render.yaml`.
5. Confirm `/_stcore/health`.
6. Run a targeted document query and verify retrieved evidence.
