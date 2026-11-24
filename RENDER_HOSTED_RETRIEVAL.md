# Hosted OpenSearch Retrieval Profile

The hosted EvidenceFlow application uses OpenSearch sparse-first retrieval. The container can connect to an externally provisioned OpenSearch endpoint; the endpoint must already provide the sparse-search plugin/model capability if `OPENSEARCH_NEURAL_SPARSE=true` is selected.

## Hosted Render configuration

The root `render.yaml` currently defines a single Streamlit service named `evidenceflow-langgraph`.

Verified public application URL:

`https://langgraph-rag-hdkn.onrender.com`

The Streamlit health route is:

`https://langgraph-rag-hdkn.onrender.com/_stcore/health`

The hosted manifest sets `OPENSEARCH_NEURAL_SPARSE=false` by default and configures the Jina reranking backend. It declares the following external settings as needed:

```text
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
```

OpenRouter, web research, optional primary-source tools, and the shared Gateway also depend on their respective server-side environment settings. Keep all credentials in Render environment settings, not in the repository or browser.

## Retrieval architecture

```text
Streamlit / LangGraph
        |
        +--> OpenSearch lexical retrieval
        |       |- BM25
        |       |- exact / phrase / fuzzy
        |       `- metadata + field boosts
        |
        +--> optional neural-sparse retrieval
        |       `- sparse token-weight retrieval
        |
        +--> RRF fusion
        +--> configured reranker (Jina on hosted Render)
        +--> evidence verification / guardrails
        `--> LLM synthesis
```

Dense-vector k-NN is not the foundation of this profile. Classical inverted-index lexical retrieval remains the baseline; neural-sparse retrieval is an optional additional signal and is disabled by default in the current Render manifest.

## Local profile

The local Docker/development configuration may use its configured local cross-encoder backend. Do not assume the local reranker choice is the same as the hosted Render backend: the hosted manifest explicitly selects Jina.

## Free-hosting caveat

The Streamlit service does not provision OpenSearch or reranking infrastructure. A working hosted deployment depends on separately provisioned OpenSearch capacity and valid external credentials. Render service health alone does not prove ingestion, retrieval, citation verification, or provider inference. Validate the full workflow with synthetic documents before calling the application demo-ready for live use.