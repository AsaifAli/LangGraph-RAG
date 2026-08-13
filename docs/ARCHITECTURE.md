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
  |      |      Dense + BM25
  |      |          -> RRF
  |      |          -> Cross-encoder rerank
  |      |          -> Self-correction
  |      |          -> Chunk analysis
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

## Why this design

The project deliberately uses a custom LangGraph `StateGraph` instead of a
heavier autonomous-agent framework for the primary chat path. The workflow is
bounded and predictable: route, retrieve/research, analyze evidence, synthesize,
and verify. `Send` fan-out provides parallel chunk analysis without exposing an
unnecessary subagent abstraction to the user.

## Retrieval

The knowledge-base path uses Qdrant hybrid retrieval: dense embeddings and a
BM25 sparse leg are fused with RRF, then candidates are reranked locally with a
cross-encoder. Retrieval is document-scoped and tenant-scoped; an empty document
scope is fail-closed and returns no knowledge-base matches.

## Evidence and QA

The model receives evidence IDs for retrieved chunks. The final answer's
`[EVID: Ex]` markers are checked against a closed registry built from the exact
chunks supplied to the model. Invalid evidence IDs and document mismatches are
removed in verified mode. Deterministic number/date checks add a second guard
for material figures. These checks are deliberately described as citation QA,
not as a claim that an LLM has been mathematically proven truthful.

## Persistence

LangGraph's SQLite checkpointer stores conversational state. Streamlit keeps a
separate JSON session index for UI rendering. Runtime state is intentionally
excluded from source control and is created by the application/container.
