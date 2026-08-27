# Hosted OpenSearch Retrieval Profile

The current application is designed around OpenSearch sparse-first retrieval. The container can connect to any hosted OpenSearch endpoint; the endpoint must already provide the sparse-search plugin/model capability if `OPENSEARCH_NEURAL_SPARSE=true` is desired.

## Runtime architecture

```text
Streamlit / LangGraph
        |
        +--> OpenSearch lexical retrieval
        |       |- BM25
        |       |- exact / phrase / fuzzy
        |       `- metadata + field boosts
        |
        +--> optional neural-sparse retrieval
        |       `- inverted-index sparse representation
        |
        +--> RRF fusion
        +--> local cross-encoder reranking
        +--> evidence verification / guardrails
        `--> LLM synthesis
```

No dense k-NN vector retrieval is used by this profile. OpenSearch neural-sparse is optional; the application falls back to lexical retrieval if sparse-model provisioning is unavailable. OpenSearch documents neural sparse search as an inverted-index-based alternative to dense retrieval, and its current documentation recommends doc-only mode for a good performance/relevance balance.

## Environment variables

```
OPENSEARCH_URL=<hosted OpenSearch endpoint>
OPENSEARCH_USERNAME=<username>
OPENSEARCH_PASSWORD=<password>
OPENSEARCH_VERIFY_CERTS=true
OPENSEARCH_NEURAL_SPARSE=true
RERANK_ENABLED=true
JINA_API_KEY=<optional, only if using a hosted reranker>
```

The default application uses the local cross-encoder via `fastembed`, so no Jina API key is required for reranking.

## Free-hosting caveat

OpenSearch with local neural-sparse model hosting is resource-intensive. The local Docker profile is the primary reproducible path. A truly free hosted Streamlit service still needs a separately hosted OpenSearch endpoint; the public application should not silently assume that a free managed OpenSearch cluster exists.
