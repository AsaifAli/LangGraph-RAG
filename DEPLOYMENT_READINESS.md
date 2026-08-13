# Deployment Readiness

Status: Phase 1 hardened; resource-intensive deployment.

Cloud shape: Streamlit web service + external Qdrant + external LLM provider. The local embedding/reranker models and CPU PyTorch footprint make Render Free a poor target; Phase 2 will decide between a lighter hosted retrieval stack and a higher-memory service.
