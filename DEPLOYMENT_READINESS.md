# EvidenceFlow deployment readiness

## Deployment topology

The repository currently includes a single Render web service, `evidenceflow-langgraph`, defined in `render.yaml`. It runs the Streamlit application from the repository's Render Dockerfile and exposes Streamlit's `/_stcore/health` health endpoint.

The hosted application depends on an externally provisioned OpenSearch endpoint and externally supplied model/research credentials. The Render Blueprint declares these as environment variables; it does not provision an OpenSearch cluster or those third-party accounts.

## Required hosted configuration

Set the following values in the EvidenceFlow Render service:

- `OPENSEARCH_URL`
- `OPENSEARCH_USERNAME`
- `OPENSEARCH_PASSWORD`
- `JINA_API_KEY` when hosted Jina reranking is enabled
- `OPENROUTER_API_KEY`
- `TAVILY_API_KEY` when web research is enabled
- `KATZILLA_API_KEY` when the optional integration is enabled
- `EVALUATION_HARNESS_INGEST_TOKEN` for best-effort, content-free activity reporting

The Render manifest sets `OPENSEARCH_NEURAL_SPARSE=false` by default. The classical OpenSearch lexical/BM25 path is the baseline. Optional neural-sparse retrieval requires a compatible OpenSearch deployment and model pipeline. Do not enable it until that capability is provisioned and tested.

The shared Portfolio LLM Gateway settings are configured in the Render manifest through `BACKEND=openrouter` and the project handoff integration; the website remains the BYOK credential owner. Do not place user provider keys in the EvidenceFlow frontend or commit secrets.

## Health and readiness checks

After the Render deployment is active, verify:

1. `https://<evidenceflow-host>/_stcore/health` returns the Streamlit health response.
2. The Streamlit page loads and can reach the configured OpenSearch endpoint.
3. A small synthetic document can be ingested and retrieved.
4. A query returns citations that resolve to the current turn's evidence.
5. Gateway handoff succeeds when launched from the website.
6. Telemetry is accepted by the shared Engineering Assurance service when the two ingestion-token configurations match.

A configured Render Blueprint does not prove that the external OpenSearch service, secrets, or live user workflow are ready. As of this documentation update, the previously used `evidenceflow-langgraph.onrender.com` hostname did not return the expected Streamlit health route during a live check; confirm the actual Render service URL and current deployment state before treating the hosted instance as available.

## Local reproducibility

The local Docker profile remains the reproducible development path:

```bash
docker compose up --build
```

Use synthetic/demo material until external access controls, tenant isolation, retention, and data-handling requirements for real documents have been reviewed.
