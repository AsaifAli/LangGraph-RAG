# Architecture

## Request flow

```text
User
  |
  v
Streamlit UI
  |
  v
LangGraph StateGraph
  |
  +--> Route / tool calling
  |      |
  |      +--> Knowledge Base
  |      |      Lexical retrieval
  |      |        -> optional neural sparse
  |      |        -> RRF
  |      |        -> cross-encoder rerank
  |      |        -> bounded agentic query expansion
  |      |        -> chunk analysis
  |      |
  |      +--> Web Research
  |             Search -> evidence -> synthesis
  |
  v
Evidence-aware synthesis
  |
  v
Deterministic citation verifier
  |
  v
Citation QA + numeric/date support checks
  |
  v
Final answer + citations + telemetry
```

## Retrieval strategy

The knowledge base uses a sparse-first design. OpenSearch performs classical lexical retrieval with BM25, exact/phrase/fuzzy matching, field boosting, and tenant/document filtering. When enabled, OpenSearch neural sparse retrieval adds learned sparse semantic signals while continuing to use an inverted-index-based search path. The candidate lists are fused with reciprocal rank fusion and optionally reranked locally with a cross-encoder.

For difficult queries, the LangGraph retrieval layer can ask the LLM to generate a small number of alternative lexical queries. Those alternatives are then executed by OpenSearch; the LLM is not used as a 100-document relevance ranker. The loop is capped in code to protect latency and token cost.

## Evidence and QA

Retrieved chunks carry stable evidence IDs and source metadata. The final answer's `[EVID: E<n>]` markers are checked against the exact evidence set supplied to the model. Guardrails treat retrieved documents and web results as untrusted data and prevent retrieved text from authorizing tool behavior.

## Persistence

LangGraph's SQLite checkpointer stores conversational state. Streamlit keeps a separate JSON session index for UI rendering. Runtime state is intentionally excluded from source control.
