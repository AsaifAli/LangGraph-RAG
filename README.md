# EvidenceFlow API — Verified Sparse-First RAG & Research

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

### 3. Bounded entity-graph expansion

EvidenceFlow also has a lightweight GraphRAG-style retrieval pass over an implicit passage/entity graph stored in the same OpenSearch index. At ingestion, deterministic extraction adds normalized entity candidates, acronyms and standards to each chunk's entity_terms field. Passages sharing those terms are connected through entity nodes. After the sparse first pass, retrieval can traverse up to two hops and add at most six graph-neighbour chunks before the existing cross-encoder reranker selects the final evidence.

Graph expansion reuses the same verified owner, tenant and explicit document-ID allow-list as ordinary search at every hop. It cannot broaden access to another conversation's files. This is an entity co-occurrence graph, not a semantic subject-predicate-object ontology or a separate graph database. New uploads receive entity terms automatically. Existing indexes can be enriched without re-uploading the source files: run python scripts/backfill_graph_entities.py for a dry-run count, then python scripts/backfill_graph_entities.py --apply to update the configured tenant index. The script can optionally be limited with --owner-scope or repeated --document-id arguments; it writes only derived entity_terms, not passage text.

Configure with GRAPH_RAG_ENABLED, GRAPH_RAG_MAX_HOPS, GRAPH_RAG_MAX_ENTITIES and GRAPH_RAG_MAX_CHUNKS in .env. The default traversal is capped at two hops, twelve entity terms per hop and six additional chunks.

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

The fused candidates can then be passed to a **cross-encoder reranker**. In the deployed configuration this can use the hosted Jina AI reranker; local development can use the configured local reranking backend. After reranking, EvidenceFlow can append a small number of neighbouring chunks from the same document to restore surrounding paragraph/table context that a narrow match may omit. These context-only chunks retain their own page/chunk provenance, are visibly distinguished in metadata, and are constrained by the exact same owner, tenant, and active-document allow-list.

Reranking is an enhancement rather than a hard dependency: if the reranking service fails, EvidenceFlow can return the fused OpenSearch candidates instead of taking down the retrieval request.

## Core retrieval stack

- **OpenSearch inverted index** for vectorless lexical retrieval
- **BM25** for classical sparse relevance scoring
- **OpenSearch neural-sparse retrieval** as an optional semantic sparse signal
- **RRF** for combining independently ranked sparse retrieval results
- **Jina AI cross-encoder reranking** for final candidate ordering when enabled
- **Bounded agentic query expansion** for weak first-pass retrieval
- **Adjacent-chunk context expansion** for surrounding paragraph and table continuity
- **Fail-closed document scoping**: an empty authorised allow-list yields no results, never an owner-wide search
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

## API-first deployment

EvidenceFlow is deployed as an API service. It does not ship a standalone product UI. The OpenAPI schema and interactive API reference are served at /openapi.json and /docs; health and readiness are available at /health and /ready. The authenticated Studio workspace provides a protected API guide and sends requests through its same-origin, client-scoped proxy.

The document ingestion API supports PDF, legacy DOC, DOCX, XLS, XLSX, XLSM, XLSB, ODS, PPTX, ODP, ODT, RTF, TXT, Markdown, CSV, TSV, JSON, XML, YAML, TOML, INI, logs, HTML, EML, SQL, common source/config text and PNG/JPG/WEBP/BMP/TIFF images. Text and tables are extracted into attributable passages; visual analysis for image/scanned content remains subject to the existing multimodal configuration. The extraction pipeline applies upload-size, archive-expansion, archive-path and format-signature validation.

### Secure portfolio demo session

When EvidenceFlow is launched from the AI Engineering Studio client workspace, the API receives a short-lived, single-use handoff code. The Studio proxy removes launch credentials from the browser URL, exchanges the code server-side with EvidenceFlow and the shared Portfolio LLM Gateway, and stores the resulting session only in client-scoped HttpOnly cookies and the backend session store. The legacy portfolio_llm_session JWT query parameter is rejected.

Configure `PORTFOLIO_SESSION_GATEWAY_URL` in the EvidenceFlow API service environment with the shared Portfolio LLM Gateway root URL, for example `https://portfolio-llm-gateway.onrender.com`. Do not point this setting at the local generic LiteLLM service. The regular `LLM_GATEWAY_URL` continues to configure the OpenAI-compatible inference endpoint.

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
RETRIEVAL_CONTEXT_WINDOW_CHUNKS=1
RETRIEVAL_CONTEXT_MAX_EXTRA_CHUNKS=6
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

`Python` · `LangGraph` · `OpenSearch` · `BM25` · `Neural Sparse` · `RRF` · `Jina AI` · `FastAPI` · `Docker` · `OpenRouter` · `Tavily`

## Repository

**GitHub:** https://github.com/AsaifAli/LangGraph-RAG


## Internal engineering assurance

EvidenceFlow reports a small, content-free event when a chat turn completes or fails. The event contains duration, retrieval counts, whether web/knowledge sources were used, and the evidence-audit result. User questions, answers, source excerpts, document IDs/names and uploaded files are never sent to the evaluation service.

Configure these server-side variables on the EvidenceFlow deployment:

- `EVALUATION_HARNESS_URL=https://ai-engineering-evaluation-harness.onrender.com`
- `EVALUATION_HARNESS_INGEST_TOKEN` — the EvidenceFlow-specific token configured by the harness operator.

Telemetry is best-effort. If the URL or token is missing, or the evaluation service is unavailable, EvidenceFlow continues its normal workflow without blocking the user. The website links to the shared read-only engineering assurance dashboard; this app does not run a separate harness service or require an additional provider key.


### Runtime evaluation signals

When a chat turn completes, EvidenceFlow sends a separate `evaluation.completed` event containing only deterministic, scalar checks computed by the application:

- Citation coverage and citation verification ratios.
- Verified/partial grounding, trust-review, security-blocked and abstention flags.
- Numeric-claim support.
- Verified/proposed evidence counts.
- Retrieved chunk count, graph-expanded chunk count and web-source count.
- Guardrail finding count and blocked flag.

These values describe deterministic runtime checks. They are **not** RAGAS/DeepEval LLM-judge scores and are not mixed into the replay fixture score.

### Optional live RAGAS judging

Live RAGAS judging is disabled by default. To enable the first supported judge pass:

1. Install the optional evaluator dependencies with `pip install -r requirements-evaluation.txt`.
2. Set `EVALUATION_RAGAS_ENABLED=true` on the EvidenceFlow API service.
3. Ensure the existing Gateway session handoff and `LLM_GATEWAY_URL` configuration are working.

The opt-in background evaluator uses the current request's short-lived Gateway session to score Faithfulness and Context Precision without reference. It does not delay or change the user-facing answer. Answer Relevancy is not enabled until a separate embedding model is explicitly configured; the evaluator does not choose one implicitly.

RAGAS receives the user's current question, the generated answer and the retrieved passages only for the evaluator call through the configured Gateway. The Evaluation Harness receives only normalized numeric scores and optional version labels. No prompts, answers, retrieved contexts, source names, document IDs, Gateway tokens or credentials are sent to the harness. The feature is best-effort and disabled by default; judge requests can incur provider usage/cost and should only be enabled after the configured Gateway/provider's data-processing terms are reviewed for the intended data.

Optional lineage variables are `EVALUATION_EXPERIMENT_ID`, `EVALUATION_DATASET_VERSION`, `EVALUATION_PROMPT_VERSION` and `EVALUATION_CONFIG_VERSION`. The model label comes from the active Gateway session. Without a configured experiment ID, runtime experiment comparisons group on the available model/prompt/config lineage.

## API contract and local development

EvidenceFlow is an API, not a separate chat interface. The authenticated AI Engineering Studio workspace displays the API contract and sends requests through the same-origin client-scoped proxy. Direct service documentation is available from /docs and /openapi.json.

The main route groups are:

- Sessions: GET /api/v1/session, POST /api/v1/session/exchange, POST /api/v1/session/revoke.
- Conversations: GET/POST /api/v1/chats, GET/DELETE /api/v1/chats/{thread_id}.
- Documents: GET/POST /api/v1/documents, DELETE /api/v1/documents/{document_id}.
- Research: POST /api/v1/chats/{thread_id}/messages (Server-Sent Events).

The research stream emits status, task, token, final and error events. Task events report a bounded lifecycle (running, completed or failed) for approved workflow steps such as request planning, knowledge retrieval, chunk analysis, document summarization, web/primary-source research and answer synthesis. These are operational progress signals only: prompts, hidden chain-of-thought, raw tool output and private intermediate model content are not exposed. Final responses retain the existing evidence-audit gate; answer tokens are released only after the independent audit passes.

Document upload supports PDF, DOC/DOCX, XLS/XLSX/XLSM/XLSB, ODS, PPT/PPTX/ODP, ODT, RTF, TXT, Markdown, CSV, TSV, JSON, XML, YAML, TOML, INI, LOG, HTML, EML, SQL, common source/config text files and PNG/JPG/WEBP/BMP/TIFF images. Workbook extraction preserves sheet names, headers and cell values. Image and scanned-PDF visual descriptions remain subject to the existing multimodal configuration.

Run the API locally from the repository root:

    uvicorn app.api_main:app --reload --port 8000

Both Dockerfiles start the API on port 8000. Docker Compose maps it to port 18000 locally. The client workspace calls EvidenceFlow through the Studio same-origin proxy; provider credentials and reusable Gateway bearers remain server-side.

The current LangGraph checkpointer remains SQLite in the service's chat_history data directory. On hosting with ephemeral filesystems, local checkpoint files can be lost when the service is replaced. This does not claim cross-deploy checkpoint durability; a durable shared checkpointer is a separate infrastructure decision.
