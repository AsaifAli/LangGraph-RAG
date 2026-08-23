# EvidenceFlow — Verified RAG & Research

> **LangGraph-based agentic RAG for research and document intelligence, built around evidence provenance, citation verification, safe retrieval, and fail-closed answers.**

EvidenceFlow is designed around one principle:

> **An AI system should not claim evidence it cannot verify.**

## What changed in the portfolio-ready release

The core agentic RAG architecture remains intact while the product now adds a focused **trust layer**:

- **Agent-decides routing:** users do not select manual retrieval modes; the graph decides whether to use the knowledge base, web research, both, or no retrieval.
- **Evidence-first answers:** source metadata, routes, verification state, conflicts, and citations are surfaced at the answer level.
- **Citation verification:** generated citations are checked against the exact evidence set.
- **Evidence sufficiency:** the system can distinguish verified, review-needed, unverified, blocked, and no-evidence outcomes.
- **Retrieved-content safety:** uploaded documents and web results are treated as **untrusted evidence, never as executable instructions**.
- **Prompt-injection defense:** suspicious retrieved instructions are detected and isolated from agent behavior.
- **Tool/MCP safety boundaries:** untrusted content cannot directly authorize arbitrary tool behavior.
- **Output guardrails:** obvious secret leakage and unsafe output patterns can be blocked before the answer is returned.
- **Fail-closed behavior:** when evidence is insufficient or conflicts cannot be resolved, the product tells the user instead of presenting an unverified answer as fact.
- **Focused trust harness:** targeted regression tests cover guardrails, citation validity, conflicts, unsafe content, and related evidence controls.

## Core retrieval stack

- LangGraph agentic routing
- Knowledge-base / web / hybrid routing
- Dense + sparse retrieval
- Reciprocal Rank Fusion
- Qdrant vector search
- BM25 sparse retrieval
- Cross-encoder reranking
- Persistent conversation state
- Whole-document evidence handling
- Evidence registry with turn-scoped IDs
- Citation coverage and validity checks
- Numeric/date evidence checks
- Evidence conflict detection

## Architecture

```text
User Question
      ↓
LangGraph Agentic Router
      ├── Knowledge Base
      ├── Web Research
      ├── Both
      └── Direct Response
             ↓
       Hybrid Retrieval
             ↓
          RRF Fusion
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

## Trust model

EvidenceFlow separates two questions:

**Safety:** can the interaction safely proceed?  
**Validity:** is the answer actually supported by evidence?

The product does not expose evaluation-framework scores as a user-facing dashboard. Instead, users see meaningful states such as:

- `Evidence verified`
- `Limited evidence`
- `Conflicting evidence`
- `Unable to verify`
- `Unsafe content blocked`

## Local development

```bash
docker compose up --build
```

The Streamlit UI and supporting services are defined by the project compose/deployment configuration.

## Testing

Run the project test suite and the focused trust-layer tests in CI. The trust layer is intentionally lightweight: it is designed to protect and validate EvidenceFlow rather than turn the project into a generic LLM evaluation platform.

## Portfolio positioning

**Evidence-centered agentic RAG** — adaptive retrieval, provenance, citation validation, safe handling of untrusted sources, and fail-closed behavior.
