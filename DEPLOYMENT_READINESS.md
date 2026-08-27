# Deployment Readiness

Status: local Docker profile is the canonical reproducible deployment.

Architecture: Streamlit + LangGraph + OpenSearch sparse-first retrieval + local cross-encoder reranker + external LLM provider.

OpenSearch neural-sparse is the semantic retrieval enhancement. If the OpenSearch ML plugin/model cannot be provisioned, the application remains functional using lexical BM25/exact/phrase/fuzzy retrieval and bounded agentic query expansion.

Hosted free-tier deployment is intentionally not claimed here: OpenSearch with local model hosting needs materially more compute than a small Streamlit container, so a hosted deployment requires a suitably provisioned external OpenSearch service.
