# Free deployment: Streamlit Community Cloud

This project is prepared for Streamlit Community Cloud rather than a Docker Space.

Runtime path:
- Streamlit UI
- local CPU embedding + sparse encoder + reranker
- external Qdrant Cloud
- external OpenRouter LLM

Required secrets/variables in the Community Cloud Advanced settings:
- `OPENROUTER_API_KEY` (secret)
- `QDRANT_URL` (variable)
- `QDRANT_API_KEY` (secret)
- `BACKEND = openrouter`
- `OPENROUTER_BASE_URL = https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL = openai/gpt-4o-mini`
- `TENANT_SCHEMA = portfolio_demo`
- `PLATFORM_TENANT_ID = langgraph_rag`
- `RAG_SELF_CORRECT_ENABLED = true`

Optional:
- `TAVILY_API_KEY` for live web research.

The app's local chat SQLite state and uploaded files are ephemeral on hosted restarts; durable knowledge-base vectors live in Qdrant.
