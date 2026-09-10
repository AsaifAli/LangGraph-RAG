# EvidenceFlow — Verified Sparse-First RAG & Research

> **LangGraph-based agentic RAG for research and document intelligence, built around sparse-first retrieval, evidence provenance, citation verification, and fail-closed answers.**

EvidenceFlow is a portfolio-ready AI system designed around a simple principle:

> **An AI system should not claim evidence it cannot verify.**

## Why this project

Most RAG demos stop at *retrieve → generate*. EvidenceFlow focuses on the engineering problems that matter when retrieval and agents have to be trustworthy:

- Which sources should the system use?
- Is the retrieved evidence safe to consume?
- Does the final answer actually cite the evidence that supports it?
- What happens when sources conflict or evidence is missing?
- How should the system behave when verification fails?
- Can useful retrieval work without a dense-vector database?

The result is a **trust-oriented, sparse-first agentic RAG pipeline** rather than a conventional vector-search chatbot.

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

## Retrieval architecture

EvidenceFlow is deliberately **sparse-first and vector-database-free**. The knowledge-base retrieval layer does **not** depend on dense embeddings or k-NN vector search.

Instead, it uses OpenSearch's indexed retrieval capabilities in two complementary sparse forms:

### 1. Classical lexical retrieval

The baseline retrieval path uses OpenSearch's **inverted index** and standard text search to retrieve passages using lexical evidence.

- `text` fields are indexed for document content and metadata such as titles, headings, sections, and keywords.
- **BM25** provides relevance scoring for lexical matches.
- Exact, phrase, fuzzy, field-boosted, and metadata-filtered queries are used where appropriate.
- Tenant and document filters constrain retrieval to the correct knowledge scope.

This gives the system a genuine **vectorless retrieval path**: relevant passages can be found through the inverted index without generating dense embeddings or maintaining a vector database.

### 2. Optional neural-sparse retrieval

The system can additionally enable OpenSearch **neural-sparse retrieval**. During ingestion, an OpenSearch sparse-encoding pipeline generates token-weight representations from the passage text and stores them in a `rank_features` field (`passage_embedding`).

These representations are used for sparse relevance matching inside OpenSearch; this is **not dense-vector k-NN retrieval**.

If neural-sparse model deployment is unavailable, the system falls back to the classical lexical path rather than making neural retrieval a hard dependency.

### Retrieval flow

```text
                    User Query
                        ↓
             ┌──────────┴──────────┐
             ↓                     ↓
      Classical Lexical      Neural-Sparse
          Retrieval             Retrieval
             ↓                     ↓
      Inverted Index        Sparse Token Weights
             ↓                     ↓
            BM25            OpenSearch Sparse Search
             └──────────┬──────────┘
                        ↓
                 RRF Result Fusion
                        ↓
              Cross-Encoder Reranking
                        ↓
                  Evidence Set
```

The important architectural distinction is that **dense-vector retrieval is not the foundation of EvidenceFlow**. The classical inverted-index/BM25 path works independently, while neural-sparse retrieval is an optional additional signal.

## Agentic retrieval and self-correction

EvidenceFlow does not immediately invoke an LLM to rewrite every search query. Retrieval starts with a bounded, deterministic first pass.

When that first pass is weak, an LLM planner can generate a small number of alternative queries using likely document vocabulary, synonyms, exact phrases, abbreviations, or domain terminology. Those alternatives are sent back through the same OpenSearch retrieval layer.

```text
Initial Query
     ↓
Sparse OpenSearch Retrieval
     ↓
Enough Evidence?
   ├── Yes → Continue
   └── No
        ↓
   LLM Query Planner
        ↓
 Alternative Lexical Queries
        ↓
 OpenSearch Retrieval Again
        ↓
      RRF Fusion
        ↓
      Reranking
```

The process is bounded by a hard round limit, keeping query expansion from becoming an uncontrolled tool/LLM loop.

## Retrieval ranking

Retrieved candidates are combined with **Reciprocal Rank Fusion (RRF)** when multiple retrieval signals are available. RRF operates on the independently ranked candidate lists rather than requiring their raw scores to be directly comparable.

The fused candidates can then be passed to a **cross-encoder reranker**. In the deployed configuration this can use the hosted Jina AI reranker; local development can use the configured local reranking backend.

Reranking is an enhancement rather than a hard dependency: if the reranking service fails, EvidenceFlow can return the fused OpenSearch candidates instead of taking down the retrieval request.

## Core retrieval stack

- **OpenSearch inverted index** for vectorless lexical retrieval
- **BM25** for classical sparse relevance scoring
- **OpenSearch neural-sparse retrieval** as an optional semantic sparse signal
- **RRF** for combining independently ranked sparse retrieval results
- **Jina AI cross-encoder reranking** for final candidate ordering when enabled
- **Bounded agentic query expansion** for weak first-pass retrieval
- **Whole-document evidence handling** for stronger source context
- **Persistent conversation state** across interactions
- **Turn-scoped evidence registry** for provenance and citation traceability

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

The retrieval architecture follows the same principle: **sparse retrieval remains independently usable**, while neural-sparse retrieval, reranking, and agentic query expansion improve recall or ranking without making the system dependent on dense-vector search.

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

Relevant sparse-retrieval settings include:

```text
OPENSEARCH_NEURAL_SPARSE=true
OPENSEARCH_SPARSE_MODEL=amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill
OPENSEARCH_SPARSE_TOKENIZER=amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1
OPENSEARCH_SPARSE_PRUNE_RATIO=0.1
LEXICAL_CANDIDATE_K=40
NEURAL_CANDIDATE_K=40
RRF_K=60
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

**Evidence-centered sparse-first agentic RAG** — inverted-index/BM25 retrieval without a dense-vector database, optional neural-sparse retrieval, bounded query expansion, RRF fusion, cross-encoder reranking, provenance, citation validation, safe handling of untrusted sources, and fail-closed behavior.

## Tech stack

`Python` · `LangGraph` · `OpenSearch` · `BM25` · `Neural Sparse` · `RRF` · `Jina AI` · `Streamlit` · `Docker` · `OpenRouter` · `Tavily`

## Repository

**GitHub:** https://github.com/AsaifAli/LangGraph-RAG
