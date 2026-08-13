# LLM Gateway

This project uses a small LiteLLM proxy as the deployment-time LLM boundary. The application talks to an internal OpenAI-compatible endpoint instead of embedding an LLM model in the project container.

## Default routing

`portfolio-free` -> OpenRouter `openrouter/free` -> LiteLLM fallback -> Gemini Flash Lite.

OpenRouter's `openrouter/free` router dynamically selects an available free model. The exact upstream model can be pinned later with `OPENROUTER_LITELLM_MODEL` without changing application code.

## Environment

Create `.env` from `.env.example` and set:

- `LITELLM_MASTER_KEY` — an internal gateway key; generate a strong random value.
- `OPENROUTER_API_KEY` — your own OpenRouter key.
- `GEMINI_API_KEY` — optional fallback key.
- `OPENROUTER_LITELLM_MODEL` — defaults to `openrouter/free`.
- `GEMINI_LITELLM_MODEL` — defaults to `gemini/gemini-flash-lite-latest`.

Never commit `.env` or provider API keys. Only `.env.example` belongs in Git.

## Local run

```bash
docker compose up --build
```

If no upstream provider key is configured, the application may start but LLM requests will fail rather than silently using a bundled model. This keeps deployment behavior explicit and avoids shipping multi-GB model weights.

## Optional self-hosted inference

Projects that already support Ollama or vLLM retain those local/self-hosted paths. They are intentionally not required by the default public deployment.
