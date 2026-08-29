# EvidenceFlow — Verified Sparse-First RAG & Research

> **LangGraph-based agentic RAG for research and document intelligence, built around adaptive retrieval, evidence provenance, citation verification, and fail-closed answers.**

EvidenceFlow is a portfolio-ready AI system designed around a simple principle:

> **An AI system should not claim evidence it cannot verify.**

## Why this project

Most RAG demos stop at *retrieve → generate*. EvidenceFlow focuses on the engineering problems that matter when retrieval and agents have to be trustworthy:

- Which sources should the system use?
- Is the retrieved evidence safe to consume?
- Does the final answer actually cite the evidence that supports it?
- What happens when sources conflict or evidence is missing?
- How should the system behave when verification fails?

The result is a **trust-oriented agentic RAG pipeline** rather than a simple vector-search chatbot.

## What the agent does

The user asks a question and LangGraph determines the appropriate path instead of exposing manual retrieval modes.

```text
User Question
      ↓
LangGraph Agentic Router
      ├── Knowledge Base
      ├── Web Research
      ├── Both
      └── Direct Response
             ↓
       Retrieval / Research
             ↓
       RRF Result Fusion
             ↓
      Cross-Encoder Reranking
             ↓
    Evidence Safety Boundary
             ↓
       Evidence Set / Registry
             ↓
         LLM Synthesis
             ↓
      Citation / Grounding Checks
             ↓
        Output Guardrails
             ↓
        Verified Answer
```

## Core retrieval stack

- **LangGraph** for stateful agentic orchestration and routing
- **OpenSearch** for sparse retrieval and indexed document search
- **Dense + sparse retrieval** for complementary semantic and lexical signals
- **BM25** sparse retrieval
- **Reciprocal Rank Fusion (RRF)** to combine ranked retrieval results
- **Jina AI reranking** through a hosted cross-encoder reranker
- **Whole-document evidence handling** for stronger source context
- **Persistent conversation state** across interactions
- **Turn-scoped evidence registry** for provenance and citation traceability

The deployed configuration uses Jina reranking when enabled, with the retrieval candidate set reranked before synthesis.

## Trust and safety layer

EvidenceFlow treats retrieved documents and web results as **untrusted evidence**, not executable instructions.

### Prompt-injection resistance

Retrieved content is isolated from agent control flow so instructions embedded inside documents or web pages cannot simply become agent instructions.

### Citation verification

Citations are validated against the actual evidence set used for synthesis rather than trusting the model's citation text at face value.

### Evidence sufficiency

The system distinguishes between states such as:

- `Evidence verified`
- `Limited evidence`
- `Conflicting evidence`
- `Unable to verify`
- `Unsafe content blocked`

### Conflict and numeric/date checks

The trust layer includes targeted checks for evidence conflicts and for claims involving numbers and dates, helping surface cases that require additional verification.

### Fail-closed behavior

When evidence is insufficient or materially conflicting, the system can refuse to present an unsupported answer as established fact.

## Architecture principles

EvidenceFlow separates two concerns:

**Safety:** can the interaction proceed without allowing untrusted content to control the agent?

**Validity:** is the resulting answer supported by the evidence actually retrieved?

This separation keeps the trust layer practical and focused instead of turning the project into a generic LLM evaluation framework.

## Deployment

The project includes a containerized Render deployment configuration with a Streamlit health check, OpenSearch connectivity, optional web research, and configurable Jina reranking.

Key deployment settings include:

```text
RERANK_ENABLED=true
RERANK_BACKEND=jina
JINA_RERANK_MODEL=jina-reranker-v1-turbo-en
TOP_K=5
MAX_CONCURRENT_SUBAGENTS=3
MAX_RESEARCHER_ITERATIONS=3
```

Secrets such as `JINA_API_KEY`, `OPENROUTER_API_KEY`, `OPENSEARCH_PASSWORD`, `TAVILY_API_KEY`, and `KATZILLA_API_KEY` are supplied through the deployment environment rather than committed to the repository.

## Local development

The project is designed to run as a containerized application:

```bash
docker compose up --build
```

Use the included project configuration and environment variables for the required OpenSearch, LLM, reranking, and research integrations.

## Testing

The test suite and focused trust-layer regression tests cover areas including:

- citation validity
- evidence sufficiency
- evidence conflicts
- unsafe retrieved content
- guardrail behavior
- related evidence-control regressions

## Portfolio positioning

**Evidence-centered agentic RAG** — adaptive retrieval, hybrid search, RRF fusion, Jina reranking, provenance, citation validation, safe handling of untrusted sources, and fail-closed behavior.

## Tech stack

`Python` · `LangGraph` · `OpenSearch` · `BM25` · `RRF` · `Jina AI` · `Streamlit` · `Docker` · `OpenRouter` · `Tavily`

## Repository

**GitHub:** https://github.com/AsaifAli/LangGraph-RAG
