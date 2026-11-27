# EvidenceFlow API deployment readiness

## Deployment topology

EvidenceFlow is a FastAPI service deployed on the existing Render web service, evidenceflow-langgraph, defined in render.yaml. The service is API-only: the Render image does not build or serve a React or Streamlit application.

Hosted API: https://langgraph-rag-hdkn.onrender.com
API reference: https://langgraph-rag-hdkn.onrender.com/docs
OpenAPI schema: https://langgraph-rag-hdkn.onrender.com/openapi.json
Health: https://langgraph-rag-hdkn.onrender.com/health
Readiness: https://langgraph-rag-hdkn.onrender.com/ready

The client workspace uses the website's same-origin API proxy. Workspace authentication, project-scoped Gateway exchange, and EvidenceFlow's per-client HttpOnly session cookie remain server-side. The API never exposes a reusable Gateway bearer to browser code.

## Live service check — 27 November 2026

The hosted service was checked directly during this change:

- /health returned HTTP 200.
- /ready returned HTTP 200.

This confirms that the HTTP service and its readiness dependencies responded at the time of the check. It does not, by itself, prove a complete upload, indexing, retrieval, citation-verification, or research workflow. Those still require a synthetic end-to-end test against the live workspace and configured external services.

## Required hosted configuration

The Render manifest declares the externally managed settings. The service depends on:

- OpenSearch endpoint and credentials.
- Jina reranking key when hosted reranking is enabled.
- Portfolio LLM Gateway URL and its identity-validation configuration.
- OpenRouter or other configured inference provider credentials.
- Tavily and optional primary-source credentials when those tools are enabled.
- The EvidenceFlow telemetry token when content-free evaluation events are enabled.

Render does not provision OpenSearch or third-party provider accounts. Do not copy user BYOK keys into the EvidenceFlow service environment; client provider credentials remain owned by the shared Portfolio Gateway.

## Document ingestion

The API validates upload size, file signatures and Office/ODF archive boundaries before parsing. Supported types are listed at the service root and in the OpenAPI/Studio API reference. Spreadsheet extraction includes XLS, XLSX, XLSM, XLSB and ODS; presentations include PPTX and ODP.

Use synthetic material for validation. Check that a spreadsheet's sheet names, headers and cell values are present in indexed evidence before testing a query that depends on a particular cell.

## Readiness checklist

1. Confirm /health, /ready, /docs and /openapi.json respond.
2. Launch EvidenceFlow API from an authenticated client workspace and verify that the session is established.
3. Upload a small synthetic PDF and workbook through POST /api/v1/documents.
4. Verify the uploaded records appear through GET /api/v1/documents.
5. Ask a question that requires a value in the workbook and inspect the cited evidence.
6. Test insufficient evidence and confirm the API abstains rather than inventing support.
7. Revoke the EvidenceFlow session and verify the per-client session is no longer active.

## Local reproducibility

Start the API with Docker Compose:

    docker compose up --build

The API listens on port 8000 in the container and is mapped to port 18000 on the local host. Use the interactive reference at /docs.
