# Hosted EvidenceFlow API — OpenSearch retrieval profile

EvidenceFlow is an API-only service hosted on the existing Render web service. It uses OpenSearch sparse-first retrieval; it does not serve a standalone product UI.

Hosted API: https://langgraph-rag-hdkn.onrender.com
OpenAPI reference: https://langgraph-rag-hdkn.onrender.com/docs
Readiness: https://langgraph-rag-hdkn.onrender.com/ready

The Render manifest sets OPENSEARCH_NEURAL_SPARSE=false by default and configures the Jina reranking backend. OpenSearch must be provisioned separately. Keep credentials in Render environment settings and never expose provider or Gateway secrets through browser code.

## Retrieval architecture

    Authenticated client workspace / API caller
             |
             v
    FastAPI API + owner-scoped session
             |
             +--> OpenSearch lexical retrieval
             |       |- BM25
             |       |- exact / phrase / fuzzy
             |       - metadata + field boosts
             |
             +--> optional neural-sparse retrieval
             |       - sparse token-weight retrieval
             |
             +--> RRF fusion
             +--> configured reranker (Jina on hosted Render)
             +--> adjacent-chunk context (same authorised document allow-list)
             +--> evidence verification / guardrails
             - LLM synthesis

Empty document allow-lists fail closed and are never widened to every document owned by the account. Context expansion fetches only neighbouring chunks from the selected source documents and cannot introduce another conversation's private attachments. The default window is one chunk on either side, capped at six extra passages per turn; set `RETRIEVAL_CONTEXT_WINDOW_CHUNKS=0` to disable it.

Dense-vector k-NN is not the foundation of this profile. Classical inverted-index lexical retrieval remains the baseline; neural-sparse retrieval is an optional additional signal and is disabled by default in the current Render manifest.

## External settings

    OPENSEARCH_URL=<hosted OpenSearch endpoint>
    OPENSEARCH_USERNAME=<username>
    OPENSEARCH_PASSWORD=<password>
    OPENSEARCH_VERIFY_CERTS=true
    OPENSEARCH_NEURAL_SPARSE=false
    RERANK_ENABLED=true
    RERANK_BACKEND=jina
    JINA_API_URL=https://api.jina.ai/v1/rerank
    JINA_RERANK_MODEL=jina-reranker-v1-turbo-en
    JINA_API_KEY=<Jina API key>
    RETRIEVAL_CONTEXT_WINDOW_CHUNKS=1
    RETRIEVAL_CONTEXT_MAX_EXTRA_CHUNKS=6

OpenRouter, web research, optional primary-source tools, and the shared Gateway also depend on their respective server-side environment settings.

## Free-hosting caveat

Render service health does not prove ingestion, retrieval, citation verification or provider inference. Validate the full workflow with synthetic documents before calling the API ready for live use. The hosted service does not provision OpenSearch or third-party credentials.
