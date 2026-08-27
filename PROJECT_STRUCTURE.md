# Project structure — LangGraph RAG chat app

A step-by-step map of `poc/langgraph_rag/`: what each file/directory is,
and where it sits in the request pipeline. For the *why* behind design
decisions (live bugs found and fixed, backend/quota notes, deepagents vs.
custom-LangGraph design tradeoffs), see `README.md` — this file is the
structural map, that one is the deep dive.

## Directory tree

One folder per concern — agent/graph, tools, retrieval, citations, config,
shared leaf modules, UI, and standalone scripts:

```
poc/langgraph_rag/
├── .env.example                  # safe configuration template; real .env is never shipped
├── requirements.txt               # core runtime dependency list
├── requirements-experiments.txt   # optional deepagents reference scripts
├── README.md                     # full write-up: decisions, bugs, design notes
├── PROJECT_STRUCTURE.md          # this file
├── Dockerfile / docker-compose.yml / .dockerignore
│
├── app/
│   └── streamlit_app.py           # chat UI over agents/custom_langgraph_poc.py
├── agents/
│   └── custom_langgraph_poc.py    # the production agent (lean LangGraph, no deepagents)
├── retrieval/                     # config, retrieval pipeline, filters
│   ├── rag_pipeline.py
│   ├── kb_filter_builder.py        # OpenSearch tenant/document filter builder
│   └── entity_filters.py           # canonical person/org entity filter conditions
├── tools/
│   └── web_search_tool.py         # Tavily-backed web search
├── citations/
│   └── verifier.py                # pure, stateless citation verifier
├── config/
│   └── loader.py                  # reads this app's own .env
├── shared/                        # leaf modules used across the tree
│   ├── constants.py                 # OpenSearch payload keys, rerank model id, error codes
│   ├── exceptions.py                # ApplicationError/StorageError
│   ├── entities.py                  # SanitizedQuery/SecureAgentRequest — real runtime isinstance checks
│   ├── query_merge.py               # merge_web_search_query
│   └── text_extractors.py           # DocumentTextExtractor (PDF/DOCX/DOC/TXT)
├── scripts/
│   ├── seed_demo_kb.py            # seeds the synthetic demo documents
│   ├── ingest_document_poc.py     # CLI to ingest an arbitrary document
│   ├── research_poc.py            # optional deep-research reference pattern
│   └── data_analysis_poc.py       # optional code-interpreter reference pattern
├── demo/                          # synthetic recruiter-friendly demo documents
│   ├── policy_2025.md
│   ├── policy_2026.md
│   └── README.md
├── docs/                           # architecture, security, evaluation, demo script
├── .github/workflows/ci.yml       # automated compile/test CI
├── Makefile                        # common test/compile/benchmark commands
├── tests/                         # pytest unit tests for pure-logic modules
│
├── fixtures/                      # bundled demo data
│   └── sample_policy_schedule.csv
├── chat_history/                  # runtime state (gitignored)
│   ├── checkpoints.db              # LangGraph checkpointer — what the MODEL remembers
│   ├── <thread_id>.json            # one file per chat — what the UI renders
│   └── uploaded_documents.json     # global document registry
└── _runs/                         # per-invocation scratch (gitignored)
    └── <pattern>_<run_id>/         # research_poc.py / data_analysis_poc.py only
        ├── retrieved/               # chunks the deepagents agent saved this run
        └── transcript.md            # CLI runs only
```

Every top-level folder (`app/`, `agents/`, `retrieval/`, `tools/`,
`citations/`, `config/`, `shared/`, `scripts/`) is a plain Python package
(`__init__.py`, no package-level logic) — cross-package imports are absolute
(`from retrieval.rag_pipeline import ...`, `from shared.constants import
...`). Each entry-point file (`app/streamlit_app.py`, everything under
`scripts/`) inserts the app root onto `sys.path` at the top before those
imports, so it resolves regardless of the invocation's working directory —
see any of those files' own top-of-file comment for why.

## Retrieval pipeline (`retrieval/rag_pipeline.py`)

Everything else in this app is built on top of this one file. It has no
agent logic of its own — it's the retrieval/ingestion/config layer
underneath the agent.

- **Config and constants**: `_get_env()` reads every tunable value from
  `.env` (model ids, tenant identity, chunk size, rerank thresholds, retry
  policy) — nothing is hardcoded in the scripts that follow.
- **Retrieval**: `retrieve_and_rerank()` — sparse-first lexical + neural-sparse OpenSearch search
  (native `RetrievalMode.HYBRID`, server-side RRF fusion) followed by a
  local cross-encoder rerank (`jinaai/jina-reranker-v1-turbo-en`).
  Guarantees a per-document fallback floor so a multi-document comparison
  can't silently starve one document to zero results.
- **Self-correction**: `retrieve_with_self_correction()` wraps the above
  with a grade → rewrite → retry-once loop (LangGraph's agentic-RAG
  pattern) — catches a bad *query*, which reranking alone can't fix.
- **Whole-document fetch**: `fetch_all_document_chunks()` — a plain OpenSearch
  payload scroll (no similarity ranking), for genuine "summarize this whole
  document" requests that top-k similarity retrieval would only partially see.
- **Chunking**: `chunk_text()` — dynamic, content-adaptive splitting for
  prose (`langchain_text_splitters.RecursiveCharacterTextSplitter`,
  recursing through paragraph → line → word → character boundaries rather
  than a fixed-size cut) — and `chunk_csv_text()` (row-packing with a
  repeated header, tabular data — a text splitter has no concept of "a
  row," so this stays purpose-built rather than switching tools).
- **Ingestion/deletion**: `ingest_document()`, `delete_document()` — both
  operate on the same OpenSearch collection `retrieve_and_rerank` reads from.
- **Model construction**: `build_langchain_model(backend)` — returns a
  ready-to-use chat model for one of two backends (`gemini-api` default,
  `huggingface`).
- **Supporting modules**: the tenant/document OpenSearch filter builder
  (`retrieval/kb_filter_builder.py`), the citation verifier
  (`citations/verifier.py`), the web-search tool + query-merge logic
  (`tools/web_search_tool.py`, `shared/query_merge.py`), and document text
  extraction (`shared/text_extractors.py`).

## Getting data into the knowledge base

- **`scripts/seed_demo_kb.py`**: seeds the two synthetic documents under fixed IDs
  so the benchmark and recruiter demo are reproducible. The content itself lives
  under `demo/`, keeping the application generic.
- **`scripts/ingest_document_poc.py`**: CLI wrapper around
  `retrieval.rag_pipeline.ingest_document()` for adding an arbitrary local
  file (`.txt`/`.md`/`.csv`/`.pdf`/`.docx`/`.doc`). The same function the
  chat UI's upload widgets call — one ingestion path, not two.

## The production agent (`agents/custom_langgraph_poc.py`)

The agent `app/streamlit_app.py` actually runs — a lean, hand-built
LangGraph `StateGraph`, no deepagents. See README "Design notes" for the
full reasoning behind this over a deepagents-based implementation.

- `route` node — one LLM call that classifies the question (KB / web / both
  / neither), decides a `kb_mode` (see below), and rewrites it into a
  self-contained query using conversation history.
- `kb_mode`: SEARCH (top-k similarity — the default, for targeted facts and
  narrow comparisons), SUMMARY (whole-document fetch, for "summarize this"/
  "what topics does it cover"), or COMPARE (whole-document fetch of two or
  more specifically named documents, for a broad comparison rather than one
  narrow attribute). SUMMARY/COMPARE both call
  `retrieval.rag_pipeline.fetch_all_document_chunks` instead of top-k similarity
  search — top-k retrieval only ever sees the passages most similar to the
  query, which silently misses most of a long document.
- `retrieve_kb` node — SEARCH mode is backed by `retrieve_with_self_correction`.
- `analyze_chunk` — one node invocation per retrieved chunk, fanned out in
  parallel via LangGraph's `Send` (the documented map-reduce pattern) — an
  isolated context per chunk, no virtual filesystem in between.
- `group_chunk_analyses` -> `summarize_document` (fanned out per document,
  same `Send` pattern) — a SECOND map-reduce level, run only when a turn's
  findings span more than one document: the documented Orchestrator-Worker
  pattern, applied to produce one clean per-document rollup before the
  final answer has to compare across documents. Skipped for the
  single-document case — no comparison to earn the extra call.
- `research_web` / `respond_direct` / `synthesize` — the web path (a real
  multi-call search-assess-refine loop, hard-capped in code at
  `POC_MAX_WEB_SEARCH_CALLS`) and final answer synthesis, converging back
  from whichever branches actually ran.
- `build_langgraph_agent()` — build once per session; `run_langgraph_turn()` /
  `astream_langgraph_turn()` — invoke many times against the same
  checkpointer + `thread_id`. A `messages` state channel
  (`langgraph.graph.message.add_messages`) carries multi-turn memory, the
  same mechanism LangGraph's own chatbot-with-memory pattern uses.
- Every LLM-calling node carries a `RetryPolicy` (3 attempts, exponential
  backoff) — added after a real Gemini 503 hit twice in a row during
  development required a full manual re-run to recover from; LangGraph's
  default retry predicate already covers provider 5xx errors, verified
  against the installed source before relying on it.
- Deliberately not included: `write_todos` planning (this workflow's shape
  is fixed, not something dynamic replanning helps) and skills-loading (see
  README "Design notes" for why lazy skill discovery proved unreliable for
  routing decisions). Not reproduced: automatic long-conversation
  summarization (a disclosed simplification, not a hidden gap).

## Single-pattern deepagents reference scripts

Each demonstrates ONE deepagents pattern in isolation — not how
`app/streamlit_app.py` actually runs (that's `agents/custom_langgraph_poc.py`,
which has no deepagents dependency at all), and not superseded by it
either: each still answers a question the production agent doesn't.

- **`scripts/research_poc.py`** — the deep-research pattern alone: `write_todos`
  planning + `research-agent` delegation + a concurrency/iteration budget,
  against Tavily-backed web search.
- **`scripts/data_analysis_poc.py`** — narrow comparison: a `LocalShellBackend`
  code-interpreter agent answering a cross-row aggregation question over the
  bundled CSV fixture, versus what semantic chunk retrieval alone returns
  for the same question. Dev-only (arbitrary local shell/code execution).

## Chat UI (`app/streamlit_app.py`)

The only UI in this app, built entirely on `agents/custom_langgraph_poc.py`'s
build-once / invoke-many API — no agent logic lives here.

- Multi-turn memory via a disk-backed `AsyncSqliteSaver`
  (`chat_history/checkpoints.db`) — survives a server restart.
- Per-chat UI state (`chat_history/<thread_id>.json`) — separate from the
  checkpointer because it carries citation/token metadata the raw
  LangChain messages don't.
- Document scope is a UNION of two deliberately different lifetimes,
  matching the two upload entry points: sidebar-uploaded documents are
  GLOBAL (`chat_history/uploaded_documents.json`, `uploaded_docs`) —
  available from every chat, survive "New chat"/switching/deleting chats,
  removed only by their own 🗑️ button. Attach-in-chat documents are
  PER-CHAT (stored inside that chat's own `chat_history/<thread_id>.json`,
  `chat_uploaded_docs`) — visible only in the chat they were attached to,
  reset on "New chat," loaded fresh on switching, and deleted for real from
  OpenSearch when that chat is deleted. Document names (not just ids) are also
  read live every turn (`_document_names_in_scope`), so the agent's routing
  prompt can reason about a real document's name instead of a bare UUID.
- Streams token-by-token via `astream_langgraph_turn`, with a live status
  panel and expanders for citations.

## Runtime state (created at run time, not checked in)

- **`_runs/<pattern>_<run_id>/`** — one directory per deepagents run
  (`scripts/research_poc.py`/`scripts/data_analysis_poc.py` only): any
  chunks it saved under `retrieved/`, and (CLI runs only) a `transcript.md`
  summary. The production agent writes nothing here — retrieved chunks
  live only in its own in-memory graph state for the duration of one turn.
- **`chat_history/`** — the chat UI's persistent state: the checkpointer
  database, one JSON file per conversation, and the global document
  registry. Real user data — never wiped by a test run or a OpenSearch
  collection recreation without an explicit, disclosed reason.
- **`fixtures/sample_policy_schedule.csv`** — the one bundled demo file,
  used by `scripts/seed_demo_kb.py` and `scripts/data_analysis_poc.py`.

## Request flow, start to finish

1. User message arrives in `app/streamlit_app.py` (or `agents/custom_langgraph_poc.py`'s
   `--query` arg) and enters the graph as a new `HumanMessage`.
2. The `route` node makes one LLM call that decides — knowledge base, web,
   both, or neither — picks a `kb_mode` (SEARCH/SUMMARY/COMPARE), and
   rewrites the question into a self-contained query using conversation
   history.
3. If KB-bound: `retrieve_kb` either calls
   `retrieval.rag_pipeline.retrieve_with_self_correction` (SEARCH — sparse-first retrieval,
   rerank, grade/rewrite retry if the first pass looks weak) or
   `fetch_all_document_chunks` (SUMMARY/COMPARE — every chunk of the
   target document(s), no ranking).
4. Retrieved chunks fan out to one `analyze_chunk` node invocation each, in
   parallel (via LangGraph's `Send`), each returning a citation-tagged
   finding. If more than one document is represented, a second fan-out
   (`summarize_document`) rolls each document's findings up before the
   final answer has to compare across them.
5. If web-bound: `research_web` runs instead (or alongside), calling the
   Tavily search tool, possibly multiple times.
6. `synthesize` combines every branch's findings into one cited answer, once
   all of that turn's branches have completed.
7. Every turn's state — including the new `AIMessage` answer — is persisted
   to the checkpointer before returning; `app/streamlit_app.py` also saves it to
   `chat_history/<thread_id>.json` for UI rendering.
