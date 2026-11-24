# EvidenceFlow deployment readiness

## Deployment topology

The repository includes a single Render web service, `evidenceflow-langgraph`, defined in `render.yaml`. It runs the Streamlit application and exposes Streamlit's `/_stcore/health` health endpoint.

**Verified hosted URL:** https://langgraph-rag-hdkn.onrender.com

The hosted application depends on an externally provisioned OpenSearch endpoint and externally supplied model/research credentials. The Render Blueprint declares these as environment variables; it does not provision an OpenSearch cluster or third-party provider accounts.

## Live verification — November 25, 2026

The correct hosted URL was checked directly:

- `https://langgraph-rag-hdkn.onrender.com/_stcore/health` returned `ok`.
- The root workspace loaded and displayed the EvidenceFlow Agentic RAG research interface.
- The workspace displayed **OpenSearch: connected**.
- The workspace displayed OpenRouter and web search as configured.

This confirms that the hosted Streamlit application is reachable and its displayed OpenSearch connection check succeeds. It does not, by itself, prove a complete document-ingestion, retrieval, citation-verification, or long-running research workflow. Those still require a synthetic end-to-end test.

Use this hostname for the portfolio link and deployment checks. The previously checked `evidenceflow-langgraph.onrender.com` hostname was not the correct live service URL.

## Required hosted configuration

Set the following values in the EvidenceFlow Render service:

- `OPENSEARCH_URL`
- `OPENSEARCH_USERNAME`
- `OPENSEARCH_PASSWORD`
- `JINA_API_KEY` for the hosted Jina reranker
- `OPENROUTER_API_KEY`
- `TAVILY_API_KEY` when web research is enabled
- `KATZILLA_API_KEY` when the optional primary-source integration is enabled
- `EVALUATION_HARNESS_INGEST_TOKEN` for best-effort, content-free activity reporting

The Render manifest sets `OPENSEARCH_NEURAL_SPARSE=false` by default. Classical OpenSearch lexical/BM25 retrieval is the baseline. Optional neural-sparse retrieval requires a compatible OpenSearch deployment and model pipeline. Do not enable it until that capability is provisioned and tested.

The hosted manifest configures the Jina reranking backend. Confirm the Jina key and upstream quota before treating reranking as operational; the application should retain its documented fallback behavior when the reranker is unavailable.

The shared Portfolio LLM Gateway is the BYOK inference boundary. The website remains the provider-credential owner. Do not place user provider keys in the EvidenceFlow frontend or commit secrets.

## Readiness checklist

1. Confirm the Streamlit health route returns `ok`.
2. Confirm the workspace loads and OpenSearch reports connected.
3. Upload a small synthetic document and verify ingestion.
4. Ask a question that requires that document and verify retrieval.
5. Inspect citations and confirm they resolve to the evidence used in the current turn.
6. Test an insufficient-evidence question and confirm the application does not fabricate supporting evidence.
7. Launch from the portfolio website and verify the one-time Gateway handoff.
8. Confirm a content-free telemetry event is accepted when the application ingestion token and the harness token map are configured.

## Local reproducibility

The local Docker profile remains the reproducible development path:

```bash
docker compose up --build
```

Use synthetic/demo material until external access controls, tenant isolation, retention, and data-handling requirements for real documents have been reviewed.