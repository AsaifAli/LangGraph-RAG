# EvidenceFlow repository map

EvidenceFlow is an API-only FastAPI service. There is no standalone React or Streamlit product UI in this repository.

    app/
      api_main.py             FastAPI routes, SSE responses, session and chat lifecycle
      api_store.py            Workspace/session persistence
      gateway_handoff.py      One-time Portfolio Gateway exchange and revocation
    shared/
      document_extraction.py  Format-specific text/table/slide extraction
      upload_security.py      Upload size, signature and archive validation
      multimodal_extractor.py PDF/image and document section extraction
      document_file_mentions.py Filename-reference detection
    retrieval/
      rag_pipeline.py         Chunking, indexing, sparse-first retrieval and reranking
    agents/
      multi_agent_orchestrator.py
      custom_langgraph_poc.py
    tests/
      API, retrieval, upload, security and evaluation regression suites

The hosted deployment uses Dockerfile.render and render.yaml. Both Dockerfiles start uvicorn app.api_main:app; the Studio client workspace consumes the API through the same-origin BFF. Interactive API documentation is served at /docs, with the OpenAPI schema at /openapi.json.
