# EvidenceFlow — Verified RAG & Research

> **LangGraph research and document intelligence with hybrid retrieval, RRF, hosted reranking, persistent conversation state, web research, and fail-closed citation verification.**

EvidenceFlow is built around a simple reliability principle:

> **An AI system should not claim evidence it cannot verify.**

## Live deployment

**Live demo:** https://langgraph-rag-hdkn.onrender.com

**Deployment:** Render Web Service

## Why this project stands out

- **Agentic routing:** the graph decides whether a request needs the knowledge base, web research, both, or neither.
- **Hybrid retrieval:** dense + sparse retrieval with reciprocal-rank fusion, followed by cross-encoder reranking.
- **Hosted Render inference:** Render uses Qdrant Cloud Inference for dense MiniLM and BM25 vectors, with hosted Jina reranking; no heavyweight neural embedding model is required in the Render application container.
- **Evidence registry:** retrieved chunks receive turn-scoped evidence IDs and generated citations are checked against the exact evidence set.
- **Fail-closed QA:** the system can abstain when evidence is missing or insufficient rather than fabricate support.
- **Persistent context:** conversation state and chat history are persisted through the configured checkpoint/session stores.

## Architecture

```text
User Query
    ↓
LangGraph Router
    ├── Knowledge Base
    ├── Web Research
    ├── Both
    └── Direct Response
           ↓
    Qdrant Hybrid Retrieval
       ├── Dense
       └── Sparse / BM25
           ↓
        RRF Fusion
           ↓
      Jina Reranking
           ↓
       Evidence Set
           ↓
      LLM Synthesis
           ↓
    Citation Verification
           ↓
      Verified Answer
```

## Retrieval stack

### Render / hosted path

```text
Document
   ↓
Qdrant Cloud
   ├── sentence-transformers/all-MiniLM-L6-v2
   ├── Qdrant BM25
   └── RRF fusion
          ↓
     Jina AI reranking
```

The Render profile keeps the application container lean by moving embedding inference to Qdrant Cloud instead of loading a local neural embedding model.

## Chat experience

The Streamlit UI is a real chat surface rather than a configuration-heavy form. The router selects the required internal path per query, while the interface exposes conversation history, document upload, retrieval/research progress, citations, and token instrumentation without asking the user to understand the internal graph topology.

## Reliability controls

- Citation verification against a closed evidence registry
- Citation coverage analysis
- Numeric/date evidence checks
- Evidence-conflict detection
- Fail-closed behavior when a knowledge-base route has no verified evidence
- Deterministic retrieval helpers and filter builders
- Persistent conversation state

## Portfolio BYOK architecture

EvidenceFlow can be launched from the portfolio with a temporary gateway session:

```text
Portfolio
   ↓
BYOK provider key
   ↓
Redis-backed session
   ↓
Short-lived JWT
   ↓
EvidenceFlow
   ↓
Portfolio LLM Gateway
   ↓
User-selected provider/model
```

The project receives the session token rather than the provider API key.

## Repository structure

```text
agents/       Production LangGraph agent
retrieval/    Qdrant retrieval, embedding, RRF and reranking
auth/         Gateway/session integration
a app/        Streamlit chat UI
citations/    Pure citation verification logic
tools/        Web research tools
shared/       Shared entities and utilities
config/       Runtime configuration
scripts/      Seeding/ingestion utilities
docs/         Architecture, evaluation and demo notes
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```

## Testing

The repository includes unit/regression coverage for retrieval, filtering, citation verification, routing, and application behavior. Run the configured CI commands locally before pushing changes.

## Honest limitations

- Citation verification validates whether a citation maps to evidence available to the turn; it does not prove that an external source itself is correct.
- Retrieval quality depends on chunking, corpus quality, filters, reranking, and query formulation.
- Web research remains dependent on the configured external search provider.
- The portfolio deployment is a demonstration environment rather than a full enterprise multi-tenant service.

## License

MIT License.
