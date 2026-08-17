
# EvidenceFlow — Verified RAG & Research

> **Portfolio project:** a LangGraph-based research system that combines hybrid dense/sparse retrieval, RRF, cross-encoder reranking, persistent conversation state, web research, and **fail-closed citation QA**. The project is designed to demonstrate not only how an answer is generated, but how the system verifies whether its evidence is actually authorized and sufficient.

## 🚀 Deployment

**Status:** Deployed

The application is deployed as a public portfolio demonstration.

**Architecture:** GitHub Actions → Docker → Cloud deployment

> Live demo access is provided selectively for evaluation/interviews.

## Why this project stands out

- **Agentic routing:** the graph decides between knowledge-base search, web research, both, or a direct response.
- **Hybrid retrieval:** dense + sparse retrieval with reciprocal-rank fusion, followed by cross-encoder reranking. The local profile runs inference in-process; the Render profile uses Qdrant Cloud Inference for the same dense MiniLM + BM25 vectors and hosted Jina reranking.
- **Evidence registry:** every retrieved chunk receives a closed-turn evidence ID; generated citations are verified against that exact evidence set.
- **Quality gate:** deterministic numeric/date support checks, citation coverage, possible evidence-conflict detection, and **abstention when a KB route has no verified evidence**.
- **Document workflows:** targeted search, whole-document summary, and cross-document comparison.
- **Production-minded foundations:** persistent LangGraph checkpoints, Docker/Qdrant, token instrumentation, health checks, tests, evaluation runner, and CI.

## Portfolio demo

See [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) for a 90-second walkthrough and [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md) for recruiter talking points. Synthetic demo policies live under [`demo/`](demo/).

---

# LangGraph RAG chat app

A standalone document-Q&A chat application: upload documents, ask questions
against them or the live web, and get cited answers with real relevance
scores — built on LangGraph (not a heavier agent framework) with a
FastAPI-free, single-process Streamlit UI for now.

The main application uses a lean hand-built LangGraph `StateGraph`. Optional
reference experiments under `scripts/` demonstrate alternative deepagents
patterns without being part of the runtime dependency path. See "Design
notes" below for the comparison and tradeoffs.

## Architecture

```
Client (Streamlit) -> LangGraph agent (route -> retrieve -> analyze -> synthesize)
                        |                              |
                        v                              v
                   Qdrant (hybrid dense+sparse        Tavily (web search)
                   retrieval + local cross-encoder
                   rerank)
```

- **`agents/`** — the production LangGraph agent (`custom_langgraph_poc.py`).
- **`retrieval/`** — the retrieval pipeline: config, chunking, embedding,
  hybrid Qdrant search + rerank, self-correcting retrieval, web search glue
  (`rag_pipeline.py`), plus the Qdrant filter builder (`kb_filter_builder.py`)
  and canonical-entity filter helpers (`entity_filters.py`).
- **`citations/`** — a pure, stateless citation verifier: checks that every
  citation an agent emits actually traces back to a chunk it was really
  given (`verifier.py`).
- **`tools/`** — the Tavily web-search tool (`web_search_tool.py`).
- **`shared/`** — leaf modules used across the tree: constants, exceptions,
  request/query entities, text extraction, query-merge logic.
- **`config/`** — environment/config loading (`loader.py`), reads this app's
  own `.env`.
- **`app/`** — the Streamlit chat UI.
- **`scripts/`** — standalone CLI entry points (seed the demo KB, ingest a
  document, the deepagents-based research/data-analysis patterns).

See `PROJECT_STRUCTURE.md` for the full file-by-file map and request flow.

Retrieval is hybrid dense+sparse with genuine server-side Qdrant Query API prefetch+RRF fusion. The default local profile uses the original MiniLM + FastEmbed BM25 + Jina ONNX reranker path. The Render profile flips only the inference location: Qdrant Cloud Inference generates the same 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` dense vector and `Qdrant/bm25` sparse vector, then Qdrant performs the same RRF fusion; the final `jina-reranker-v1-turbo-en` pass is hosted through Jina AI. See `retrieval/rag_pipeline.py` for the active backend implementation.

## Chat UI (`app/streamlit_app.py`) — a chat, not a form

An early version of this Streamlit app was a form: pick a pattern (RAG /
research / data-analysis) from a sidebar radio, pick a backend from a
dropdown, set a subagent-count slider, then click "Run." Reasonable for
*debugging* — every control mapped directly to a real function parameter —
but a bad *product* surface: a real user shouldn't have to know what a
"subagent" is, or decide in advance which internal pattern their question
needs.

Rebuilt as an actual chat: `st.chat_message`/`st.chat_input` render a
running conversation, no pattern/backend/subagent-count controls anywhere in
the primary view. Every message goes to `agents.custom_langgraph_poc`'s
`astream_langgraph_turn` — the `route` node decides for itself, per query,
whether it needs the knowledge base, the web, both, or neither. Backend
defaults to `litellm` (`rag_pipeline.DEFAULT_BACKEND`), not exposed as a
UI choice, since a real user shouldn't need to know or care which backend
answered them.

The sidebar keeps exactly three things a chat UI still needs somewhere
outside the turn-by-turn flow — none is part of asking a question:

- **🆕 New chat** — clears the conversation *and* the document scope
  (`st.session_state.messages` and `.uploaded_docs`), matching ChatGPT's own
  behavior: starting fresh doesn't carry a previous conversation's uploads
  into the new one. The underlying Qdrant data isn't deleted (uploads are
  additive, see "Document upload") — "new chat" just means those documents
  are no longer in the active query scope until re-uploaded. This only
  resets the `thread_id`, not the agent itself — see "Context management".
  The old thread isn't lost, unlike before the update below — it was
  already persisted after its own last turn, so it's one click away in the
  chat list.
- **Chats** — every past thread, titled from its first message, most
  recently updated first; clicking one restores it exactly (messages,
  citations, token counts, and document scope) via `_switch_to_chat` —
  see "Context management" for the persistence this reads from.
- **Document upload** — two entry points, both calling the exact same
  `rag_pipeline.ingest_document` (no duplicated ingestion logic) and both
  auto-ingesting on selection with no separate confirm step (attaching a
  file was already immediate in the chat input; the sidebar uploader
  originally needed an extra "Add to knowledge base" click, the one
  inconsistent step between the two paths — removed): attach a file
  directly in the chat input (`st.chat_input(accept_file="multiple",
  file_type=[...])`, for "here's a document, answer about it" in the flow of
  asking) and a sidebar `st.file_uploader` (for "load this in before I start
  asking anything," closer to ChatGPT's project-file upload). The sidebar
  uploader's widget key is bumped after each ingest so it resets to empty
  rather than silently re-ingesting the same file on the next unrelated
  rerun.

Verified live, in the actual running app (not a lower-level unit test):
Streamlit's own `AppTest` harness drove real, complete multi-turn
conversations against the real HF backend — see "What was actually tested"
for the specific transcripts (a genuine memory test: told the agent a
favorite number in turn one, asked for it back in turn two, got the right
answer, proving the checkpointer-based memory below actually works, not
just that it constructs).

## Context management — a real checkpointer, not a history hack

The first version of this passed prior turns as a hand-rolled,
tool-call-stripped `history` list into every call, because each call built a
brand-new agent from scratch — a real, disclosed compromise (see git history
of this file if you want the exact wording), not a design choice. Checked
against the actual deepagents/LangGraph mechanism for this
(`docs.langchain.com/oss/python/deepagents/context-engineering`, cross-checked
against the installed `langgraph.checkpoint.memory.InMemorySaver` rather than
trusted blind) and rebuilt properly:

- `agents.custom_langgraph_poc.build_langgraph_agent(...)` constructs the agent **once**, with
  a real `checkpointer`.
- `run_langgraph_turn` / `astream_langgraph_turn` invoke that same built agent
  **many times** against a `thread_id`, sending only the new message each
  time — LangGraph itself retrieves and prepends the prior state for that
  thread from the checkpointer, including the automatic `SummarizationMiddleware`
  compression that already ran on it (see "Token usage"). No manual history
  reconstruction, no risk of feeding one backend's tool-call format to a
  different backend mid-conversation.
- `app/streamlit_app.py` builds the agent once per browser session
  (`st.session_state.built_agent`) and reuses it across "New chat" clicks —
  "New chat" only generates a fresh `thread_id`; the checkpointer's memory is
  scoped by thread, so a new thread is a genuinely empty conversation without
  needing to rebuild tools/subagents/skills every time.

**Update — durable across restarts, plus a chat history sidebar**, per
explicit request ("can't use other chats in the UI" + "need conversation
history too"): the checkpointer is now `AsyncSqliteSaver` (`_chat_state.db`,
`langgraph-checkpoint-sqlite`), not `InMemorySaver` — a thread's real memory
survives closing the browser or restarting the server, not just calls within
one running process. A second, deliberately separate store
(`_chat_sessions.json`, written by `_persist_current_chat()`) tracks every
thread's title and rendered messages — including citation/token UI metadata
the checkpointer's raw LangChain messages don't carry — so the sidebar can
list past chats and switching one back in looks exactly like it did live,
not just "the model still remembers it." Two stores because they serve two
different readers: the checkpointer is what the *model* reads for context;
`_chat_sessions.json` is what the *UI* reads to render.

**Verified live**: sent a message, clicked "New chat" (thread count in the
sidebar went from 1 to 2, new thread empty), clicked back into the first
chat (its `thread_id` and message content were restored exactly). Then,
independently, killed the Python process entirely and started a **brand-new
one** — the past chat was still listed and its content still restored from
`_chat_sessions.json`, *and* querying the sqlite checkpointer directly
(`agent.aget_state(...)`, bypassing the UI) confirmed the real
`HumanMessage` was retrievable from `_chat_state.db` in that fresh process —
proof the model's own memory is durable, not just the UI's copy of it.
- One correctness fix this pattern required: `search_knowledge_base` used to
  close over a `document_ids` list captured once at agent-construction time.
  Wrong once the agent is built once per session — a document uploaded
  mid-conversation wouldn't have been in that frozen list. It now takes a
  `document_ids_provider` callable, re-read on every search, so
  `app/streamlit_app.py` passes `_document_ids_in_scope` (reads live
  `st.session_state`) instead of a snapshot.

**Verified live, for real** (not just construction): asked the agent *"My
favorite number is 42. Remember that."*, it acknowledged; asked *"What is my
favorite number?"* in a second, separate call on the same thread, it
answered *"Your favorite number is 42."* — full round trip through the real
checkpointer, the real HF backend, and the real Streamlit chat flow. See
"What was actually tested" for the transcript.

## Delegation — confirmed no built-in throttling exists

deepagents has no built-in, framework-enforced ceiling on how many
delegated tool calls a turn can make. Checked directly against
`docs.langchain.com/oss/python/deepagents/subagents` rather than assumed:
subagents are "synchronous by default" — the orchestrator
blocks until a delegated `task()` call finishes — and multiple subagents run
in parallel only through separate `task()` calls the *model* chooses to make
in one turn. The docs state plainly that **no explicit concurrency/throttling
configuration is documented**. This confirms rather than just repeats what
this POC's own `max_concurrent_subagents`/`max_researcher_iterations` prompt
text already implied: those numbers are advisory instructions the model can
ignore, not a harness-enforced ceiling. Nothing to fix here — the `SubAgent`
dict shape (`name`/`description`/`system_prompt`/`tools`/`model`/`middleware`/
`interrupt_on`/`skills`/`permissions`) this POC already uses matches the
documented shape exactly.

## Frontend / event streaming — real token streaming, not a spinner

The chat UI used to make one blocking `await agent.ainvoke(...)` call per
turn behind a generic "Thinking..." spinner. `astream_langgraph_turn`
replaces that with `agent.astream_events(..., version="v2")`, giving
`app/streamlit_app.py` a live `st.status()` panel ("🔍 Searching the knowledge
base...", "🌐 Searching the web...", "🤝 Delegating to a subagent...") plus
the answer streaming in token by token via `st.empty()`/`.markdown()`.

One real correction along the way: a docs-page summary described a custom
projection API (`stream.messages`, `stream.tool_calls`, `stream.subagents`).
A live dry run against the actual installed package showed this doesn't
match — `astream_events` emits the **standard LangChain event vocabulary**
(`on_chat_model_start/stream/end`, `on_tool_start/end`, `on_chain_start/end`,
...), not a deepagents-specific projection layer. The implementation is
built against the verified real event shape, not the summary.

**Verified live, for real**: the streamed "42" memory conversation above
went through this exact code path — `st.status` updates, token-by-token
`st.empty()` accumulation, and the final answer, all observed correctly in
an actual `AppTest`-driven run against the real HF backend. Tool-call status
events specifically (`on_tool_start`/`on_tool_end` firing with
`search_knowledge_base`/`tavily_search` as the event name) are standard,
well-established LangChain behavior but were not directly observed in a
live run that actually reached a tool call before hitting the credit wall —
see "What was actually tested" for exactly which parts were and weren't
observed.

## Steering / human-in-the-loop — implemented, then removed per feedback

A sidebar "🛑 Require approval before web searches" toggle was built,
passing `interrupt_on={"tavily_search": True}` to `create_deep_agent` (gating
the one tool in this agent that costs real money/quota on an external
service per call) and resuming a paused run via
`Command(resume={"decisions": [...]})`. It was verified *structurally*
against the installed `langgraph`/`deepagents` (construction, `Command`
import, `StateSnapshot.interrupts` detection all confirmed against real
package internals — see git history of this file for the detail), but real
usage of the app surfaced that the approve/reject prompt interrupting a chat
mid-flow wasn't intuitive as a UX, so it was removed entirely at the user's
request rather than kept behind a toggle nobody found clear. `agents/custom_langgraph_poc.py`
no longer sets `interrupt_on` at all — `build_unified_agent` has no
steering-related parameter, and there is no `resume_unified_turn`. If
human-in-the-loop approval is wanted again later, the removed code (last
present before this change) is a working structural reference, but the round
-trip itself was never confirmed against a real triggered interrupt before
removal — see prior revisions of this section.

## Document upload — ingest and query your own documents

Both `app/streamlit_app.py`'s chat-input file attachment and the CLI
(`ingest_document_poc.py path/to/file.txt`) call the same
`rag_pipeline.ingest_document`: extract text → chunk → embed (the same HF
embedder as everything else in this POC) → upsert into the demo Qdrant
collection, additively (`force_recreate=False`, unlike `scripts/seed_demo_kb.py`'s
fixed demo data) — uploads accumulate rather than wiping prior ones or each
other. Verified live, several times, without spending any LLM credits
(ingestion only needs the embedder, not a chat model): a real uploaded
document was chunked, embedded, written to Qdrant, and correctly retrieved
by its own `document_id` via `retrieve_and_rerank`; the original seeded demo
data was confirmed still intact and unaffected afterward. In the chat app,
uploaded documents are tracked in `st.session_state` and automatically
included in every message's `document_ids` — no manual UUID entry needed.

**Accepted types: `.txt`/`.md`/`.csv`/`.pdf`/`.docx`/`.doc`.** PDF and DOCX
text extraction is handled by `shared/text_extractors.py::DocumentTextExtractor`
(`rag_pipeline.extract_text` dispatches to it by file extension) —
pdfplumber for PDF, python-docx for DOCX (including table cells, joined
with ` | ` per row), and a LibreOffice-headless subprocess fallback for
legacy binary `.doc`. That last path needs a `LIBREOFFICE` env var pointing
at a real binary — without it, a `.doc` upload surfaces
`StorageError(LIBREOFFICE_NOT_FOUND)` rather than silently failing; `.docx`
doesn't depend on this at all and works regardless. `.txt`/`.md`/`.csv` also
route through the same extractor's `txt_bytes_to_text` (UTF-8 with a
latin-1 fallback) rather than a bare `.decode("utf-8")`.

CSV gets its own chunker
(`chunk_csv_text`), found necessary after actually testing it: plain
paragraph-packing splits on blank lines, but CSV files don't have any, so
the whole file becomes one "paragraph" that gets hard-split by raw character
window — which can cut mid-row and strips the header off every chunk after
the first, leaving bare values meaningless (`POL-10231,1000000,25000` — which
number is the sublimit, which is the deductible?). `chunk_csv_text` instead
packs whole data rows per chunk and repeats the header row in every one.
Verified directly against the repo's own
`fixtures/sample_policy_schedule.csv`: every chunk keeps the header, no row
is ever cut in half.

**Chunking caveat, disclosed rather than hidden**: `rag_pipeline.chunk_text`
is a recursive-splitter chunker (LangChain's `RecursiveCharacterTextSplitter`)
— not a structure-aware chunking service that would split on document
sections/headings for better boundaries on long structured documents.
Standing up that kind of service was out of scope; this simpler chunker is
good enough to make uploaded documents queryable.

## Token usage

Every script tracks and reports total token usage per run via
`rag_pipeline.make_token_tracker()` /
`summarize_token_usage()` — a thin wrapper around LangChain's own
`langchain_core.callbacks.UsageMetadataCallbackHandler` (not something this
POC implemented), attached via `agent.ainvoke(..., config={"callbacks": [...]})`
(or `astream_events(..., config={"callbacks": [...]})` for the streaming
path). Because the callback fires on every chat-model call within a
LangGraph run — including calls made by subagents via `task()` — this
captures the full run's token cost, not just the orchestrator's own
top-level calls. **Verified live**: a real chat turn (routed to web search,
"current Prime Minister of Nepal") reported `16790 tokens (input: 16658,
output: 132)`, rendered as a `🔢` caption under the assistant's message —
this is a real number from a real HF Inference API response, not estimated.

## Citations — knowledge base and web, both real

The chat UI shows an expander under any assistant message that used a
source: **📚 Knowledge base citations** (the retrieved chunks, each tagged
with its `[EVID: E<n>]` id, plus `citations/verifier.py`'s
`grounding_status`) when `search_knowledge_base` was used, and **🌐 Web
sources** (title + URL + Tavily relevance score for each result) when
`search_web` was used — both, if a question needed both.

`WebSearchTool` captures structured Tavily sources via
`get_captured_sources()` (`tools/web_search_tool.py`). Wiring citations in
surfaced a real bug worth noting for anyone extending this: tool closures
built once at agent-construction time can silently accumulate state
(retrieved chunks, captured sources) across an entire chat session instead
of resetting per turn if nothing explicitly clears them — a second question
in a conversation would show citations bleeding in from the first. Fixed by
resetting per-turn state (`_reset_turn_state`) at the start of every
`run_langgraph_turn`/`astream_langgraph_turn` call, including
`WebSearchTool.set_rehydration_context()` for the web-side reset.

**Verified live, for real, in the exact scenario that surfaced this whole
gap**: asked *"current Prime Minister of Nepal"* through the actual running
chat app — correctly routed to web search, answered *"The current Prime
Minister of Nepal is Balendra Shah, who took office on 27 March 2026,"* and
rendered a `🌐 Web sources (3)` expander with the three real Tavily result
URLs/titles/scores. No `📚 Knowledge base citations` expander appeared for
that turn, correctly — it was a pure web-search answer.

## Backends actually available

Two chat-model backends are supported (`build_langchain_model`/`--backend`
accept `gemini-api`/`huggingface`):

- **`gemini-api`** (the chat UI's default): Gemini over a direct Google AI
  Studio API key, this app's own `GEMINI_API_KEY`
  (`langchain_google_genai.ChatGoogleGenerativeAI`). Added to get off the
  `huggingface` backend's recurring HF Inference credit exhaustion.
  **Model id needed real live
  probing, not the obvious guess, on this specific key/project**: every
  *dated* model name tried came back `404` "no longer available to new
  users" (`gemini-2.5-flash`, `gemini-2.5-flash-lite`), and `gemini-2.5-pro`
  has a **zero** free-tier quota (`429`, `limit: 0` in the error body, not a
  transient rate limit) — only `-latest` *alias* names work on this
  project. `gemini-flash-latest` was the first found working (confirmed
  live: real tool-calling via `bind_tools(...)` → an actual `tool_calls`
  entry back) but turned out to alias to `gemini-3.6-flash`, whose free-tier
  quota is a hard **20 requests/day** (confirmed in its own quota-error
  body) — tight enough that this session's own verification testing
  exhausted it. `gemini-flash-lite-latest` — a genuinely separate quota
  bucket, confirmed live by being usable immediately after `flash-latest`'s
  cap was already hit — is what's hardcoded now; it aliases to
  `gemini-3.5-flash-lite` and was independently confirmed for tool-calling
  the same way.

  **Verified live, for real, end to end** (not just construction): asked
  *"What is the cyber liability sublimit on POL-10231?"* through the actual
  Streamlit app — no errors, `used_knowledge_base=True`, 4 chunks retrieved,
  and a correct, properly-cited answer: *"...carries a sublimit of CAD
  1,000,000 per claim, subject to a CAD 25,000 deductible [EVID: E1]...a
  related Data Breach Response sublimit of CAD 250,000...[EVID: E4]"* —
  exact match to the seeded ground truth. 42,508 total tokens tracked
  correctly under `gemini-3.5-flash-lite` in the per-model breakdown. This
  is the first backend in this entire POC session to complete a full
  real-KB-answer turn through the chat UI without hitting a credit/quota
  wall partway through.
- **`huggingface`**: `Qwen/Qwen2.5-72B-Instruct` via the HF Inference API,
  using this app's own `HF_TOKEN`. Chosen and live-verified before being
  wired in: the token's scopes were checked (`inference.serverless.write`,
  `canReadGatedRepos: false` — so an ungated model was required), then a
  real chat-completion call and a real tool-calling call confirmed a proper
  `tool_calls` response. The original default, kept working but no longer
  the chat UI's default given its own recurring credit exhaustion
  throughout testing.

**A real cross-backend bug found by switching to `gemini-api`, not
theoretical**: `agents/custom_langgraph_poc.py` assumed `BaseMessage.content` is always a
plain string (used directly in string concatenation and a citation regex).
That's true for the HuggingFace integration but not documented-safe
in general — `ChatGoogleGenerativeAI` returns `.content` as a **list of
content blocks** (`[{"type": "text", "text": "...", ...}, ...]`) even for a
plain-text answer with no multimodal content involved, which crashed with a
real `TypeError: can only concatenate str (not "list") to str` the first
time a Gemini turn actually streamed a token. Fixed with `_content_to_text()`,
applied everywhere `.content` is treated as text (streaming accumulation,
the final answer, the `[EVID: E<n>]` regex) — not just confirmed crash-free
but confirmed fully correct: the citation tags in the verified end-to-end
answer above (`[EVID: E1]`, `[EVID: E4]`) came out of exactly this code
path.

**Embedder**: `sentence-transformers/all-MiniLM-L6-v2`, run **locally**
(`langchain_huggingface.HuggingFaceEmbeddings`, backed by `sentence-
transformers`/`torch`) rather than via a hosted API — embedding calls don't
depend on any external credits or quota, independent of which chat-model
`--backend` is selected (the two are independent axes). Verified live: a
real `retrieve_and_rerank` call returned the correct top-scoring chunk (the
actual POL-10231 cyber-sublimit ground truth, score 0.6575) on the first
try. This app only ever reads/writes its own clearly-namespaced demo
collection (see below), never anything else.

`QDRANT_URL=http://localhost:6333` and Tavily's API were both reachable from
this environment; the local Qdrant instance had **zero collections** (no
ingested documents), so `scripts/seed_demo_kb.py` seeds a small, clearly-namespaced
demo collection (`poc_demo_deepagents_eval_sentence-transformers-all-MiniLM-L6-v2_embeddings`)
via the HF embedder above. This only writes to the local dev Qdrant this
repo's own `.env` already points at — it is not production and cannot
collide with any real tenant's collection naming.

## How to run

This app is fully standalone — every command runs from `poc/langgraph_rag/`
and everything it needs (code and configuration) lives inside this
directory.

```bash
cd poc/langgraph_rag
python -m venv .venv
.venv/Scripts/activate        # Windows; `source .venv/bin/activate` elsewhere
pip install -r requirements.txt

# one-time: add your tokens/keys — see "Env vars needed" below for the full list
echo "HF_TOKEN=hf_..." >> .env
echo "GEMINI_API_KEY=..." >> .env

# one-time: seed a small demo KB into the local dev Qdrant .env points at
python scripts/seed_demo_kb.py

# Production agent — one query, the model decides KB vs web vs both vs
# neither. Lean custom LangGraph, no deepagents (see "Design notes" for why).
python -m agents.custom_langgraph_poc --query "What is the cyber liability sublimit on POL-10231?"

# Deep research pattern (deepagents reference — --backend all runs huggingface + gemini-api)
python scripts/research_poc.py --query "What changed in the LangGraph 1.x release?" --backend all

# Data analysis pattern (narrow, single script)
python scripts/data_analysis_poc.py

# Ingest your own document, then query it with --document-ids <the id it prints>
python scripts/ingest_document_poc.py path/to/document.txt

# Chat UI — "just for now", per explicit ask. ChatGPT-style: no pattern/
# backend/subagent-count controls, just chat + drag a file into the message
# box to add it to the knowledge base. Always uses the production
# custom-LangGraph agent (agents.custom_langgraph_poc.build_langgraph_agent)
# — see README "Chat UI" for why.
streamlit run app/streamlit_app.py
```

### Docker (chat UI + Qdrant, one command)

```bash
cd poc/langgraph_rag
# one-time: add at least one chat-model credential to .env — see "Env vars
# needed" below. GEMINI_API_KEY (the default backend) is enough on its own.
docker compose up --build
# -> chat UI: http://localhost:18501 (not 8501 — see docker-compose.yml's
#    comment: a completely unrelated project's own Streamlit container on
#    the same Docker daemon also defaulted to 8501 in practice)
# -> Qdrant dashboard (debugging only): http://localhost:16333/dashboard
```

This starts two containers — `app` (the Streamlit chat UI, built from the
`Dockerfile` in this directory) and `qdrant` (`qdrant/qdrant:v1.16.2`,
matching the pinned `qdrant-client` version) — wired together over the
compose network; `QDRANT_URL` is overridden to `http://qdrant:6333` inside
the container regardless of what `.env` has it set to on the host (see
`docker-compose.yml`'s comment on why that override has to be an
`environment:` value, not baked into `.env`). Qdrant's REST/gRPC ports are
published on the host as `16333`/`16334`, not the Qdrant defaults
`6333`/`6334` — deliberately, so this doesn't fail to start on a machine
already running another Qdrant instance on the defaults; the `app`
container never uses the host-published ports at all, only the internal
`qdrant:6333` compose-network address. Chat history and downloaded model
weights (the local embedder + reranker, ~1GB) persist in named Docker
volumes across rebuilds — see `docker-compose.yml`'s `volumes:` section.

To run one of the CLI scripts instead of the chat UI (e.g. to seed the demo
KB or ingest a document) against the same containerized Qdrant:

```bash
docker compose run --rm app python scripts/seed_demo_kb.py
docker compose run --rm app python scripts/ingest_document_poc.py fixtures/sample_policy_schedule.csv
```

Env vars needed — all in **`poc/langgraph_rag/.env`** (gitignored;
`config/loader.py` and `retrieval/rag_pipeline.py` read only this file):
- `HF_TOKEN` — required for the `huggingface` backend and always for the
  local embedder's one-time model download.
- `GEMINI_API_KEY` — required for the default `gemini-api` backend (a direct
  Google AI Studio key).
- `QDRANT_URL` (defaults to `http://localhost:6333`), `QDRANT_API_KEY`
  (optional — local Qdrant runs without auth by default).
- `TAVILY_API_KEY` (or `WEB_SEARCH_API_KEY`) — required for `search_web`/
  `scripts/research_poc.py`.
- `LIBREOFFICE` — optional, only exercised by the rare legacy-`.doc`-via-
  LibreOffice text-extraction fallback (`.docx` doesn't need it).

Reranking runs the local `jinaai/jina-reranker-v1-turbo-en` cross-encoder
(no API key needed) — see "Backends actually available" and the reranker
note in `requirements.txt`.

LangSmith tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in
`.env` to get trace links for every run; if unset, the scripts run
identically and just print a note that tracing is off (no hard dependency on
LangSmith for the scripts to function).

## What was actually tested

Everything below is a real, live result from actual testing against this
app's real Qdrant/Tavily/HF instances — nothing here is fabricated,
including the failures.

**Backend availability:** `huggingface` (Qwen2.5-72B-Instruct) has been
inconsistently usable — sometimes blocked with `402 Payment Required`
("You have depleted your monthly included credits"), sometimes not, with no
reliable pattern beyond "short/simple exchanges succeed more often than
ones requiring a full tool-call round trip." Treat every credit-gated claim
below exactly as scoped — a success means that specific exchange happened
for real; it does not mean the account is reliably usable.

**What ran live and successfully, with real evidence:**

- **The user's own real usage of the running app, not a crafted test** —
  arguably the best evidence in this whole README, because it wasn't staged.
  Asked the real chat app *"current Prime Minister of Nepal"*: the unified
  agent correctly routed to web search (not the knowledge base — correct,
  this needed neither a company document nor a guess-a-file-path detour),
  answered accurately (*"Balendra Shah, who took office on 27 March 2026"*),
  and the UI rendered `🔢 16790 tokens (input: 16658, output: 132)` and a
  `🌐 Web sources (3)` expander with the real Tavily result URLs — closing
  the loop on the "I couldn't see total tokens or citations" gap that
  surfaced from this exact real interaction (see "Citations" and "Token
  usage" above for what was fixed).
- **Real, complete, multi-turn conversations through the actual chat app** —
  the strongest evidence in this README, and the newest. Via Streamlit's own
  `AppTest` driving the real running app against the real HF backend:
  *"hi there"* → *"Hello! How can I assist you today?"*, a full round trip
  through the new build-once/checkpointer/streaming architecture with zero
  errors. Then, on a second, separate session: *"My favorite number is 42.
  Remember that."* → acknowledged; *"What is my favorite number?"* (a
  **separate call**, same `thread_id`) → *"Your favorite number is 42."*
  This is direct, real proof the checkpointer-based memory in "Context
  management" actually works — not a mock, not inferred from construction
  succeeding.
- **A full, accurate, correct answer to the actual demo-data question** —
  earlier in the session, with an explicit instruction to call
  `search_knowledge_base`, the HF backend retrieved the right chunks and
  answered *"The cyber liability sublimit on POL-10231 is CAD 1,000,000 per
  claim, with a CAD 25,000 deductible. The policy period runs from
  2025-01-01 to 2026-01-01"* — an exact match to the seeded ground truth.
- **A real routing failure, a fix, and partial re-verification** — see "A
  real routing failure" above for the original bug (skills-based prompt
  design skipped `search_knowledge_base` and hallucinated a file path) and
  the fix (moved the routing decision into the always-loaded prompt). Two
  live re-tests of the exact previously-failing question both hit `402`
  again — but the **second** attempt's traceback showed the run had
  progressed into a `'tools'` task stage (visible in the error's task
  breadcrumb) before failing on a *subsequent* model call, which it had
  never done before the fix. That's a real, positive signal the fix changed
  behavior in the right direction, not a full confirmation of a correct
  final answer — reported at exactly that strength, no more.
- **Document upload across every supported format**, without spending any
  LLM credits (extraction + embedding only, no chat model needed): `.txt`
  (via `chunk_text`), `.csv` against the repo's own
  `fixtures/sample_policy_schedule.csv` (via `chunk_csv_text` — confirmed
  every chunk keeps its header, no row split mid-value), a real generated
  `.docx` with a table (via `DocumentTextExtractor.docx_bytes_to_text`
  — table cells correctly joined with ` | `), and a real generated `.pdf`
  (via `pdf_bytes_to_text`). Every one: extracted, chunked,
  embedded, upserted, correctly retrieved by its own `document_id`, then
  removed again, with the original seeded demo data confirmed intact
  throughout.
- **Retrieval** (`retrieve_and_rerank`, no LLM needed): `scripts/seed_demo_kb.py`
  seeded 4 real chunks into the local Qdrant via the real HF embedder, then a
  direct call retrieved them for *"What is the cyber liability sublimit on
  POL-10231?"* — top result, correctly, was the POL-10231 Cyber Liability
  chunk at cosine score 0.6575, ahead of the POL-11002 chunk (0.6331) and two
  other POL-10231 chunks (0.5521, 0.5186).
- **Web search** (`WebSearchTool` + `merge_web_search_query`): a live query
  for *"What is LangGraph?"* returned a real Tavily dual-topic (general+news)
  search, correctly picked the higher-scoring "general" topic (0.9136 vs.
  0.8437 for "news"), and returned a real synthesized answer with the
  `_WEB_PREFIX` grounding disclaimer intact. (Hit and worked around one
  Windows-only quirk along the way: `web_search_tool.py`'s debug `print()`s
  crash under Windows' default cp1252 console encoding on certain Unicode
  in Tavily responses — worked around with `PYTHONIOENCODING=utf-8`.)
- **Citation verifier**: `python citations/verifier.py`, run directly —
  `verifier self-tests PASS`.
- **Streaming and status updates**: the "42" memory conversation above went
  through the real `astream_langgraph_turn` code path — `st.status` label
  updates and token-by-token `st.empty()` accumulation both observed
  correctly in that live run.
- **API-shape corrections found by verifying instead of trusting docs
  summaries** — twice: `astream_events`'s real event vocabulary (standard
  LangChain events, not the custom projection API a docs summary described)
  and interrupt detection (`StateSnapshot.interrupts`, not the
  `result.interrupts` phrasing a different docs summary used, which doesn't
  match any real attribute `ainvoke` returns).

**Still not verified live**, and shouldn't be assumed working until they
are: a full, complete, *correct* final answer through the fixed routing
prompt (the partial "reached the tools stage" signal above is the closest
this session got); `agents/custom_langgraph_poc.py`'s mixed/neither routing
branches specifically (only the KB branch and general-chat branch were
exercised); `gemini`/`claude` output quality (never authenticated in this
sandbox at all); `scripts/research_poc.py` and `scripts/data_analysis_poc.py`'s
code-interpreter half beyond import/construction checks.

## Design notes: deepagents vs. the custom LangGraph agent

`agents/custom_langgraph_poc.py` (a lean hand-built `StateGraph`) is what
the chat UI actually runs. It replaced an earlier deepagents-based
implementation once comparison testing surfaced real, measured tradeoffs
worth recording:

1. **Token overhead.** deepagents' always-loaded scaffolding — filesystem-
   tool descriptions, `task()`/`write_todos` tool schemas, subagent
   middleware instructions — is pure overhead for a fixed-shape RAG
   workflow (route → retrieve → analyze → synthesize) that doesn't benefit
   from dynamic replanning. A live run of the deepagents version against a
   two-fact policy lookup measured 38,751 total tokens for a query the
   custom graph answers with meaningfully less overhead, since it carries
   no filesystem/planning tool schemas at all.
2. **What's worth keeping from deepagents.** Two patterns earned their
   keep and were reproduced directly in the custom graph: parallel
   per-chunk delegation (here: a plain `StateGraph` fan-out via `Send`,
   not the `task()` tool + subagent middleware) and keeping large chunk
   text out of the synthesis prompt (each chunk analyzed in its own
   isolated node call).
3. **Skill-based routing is not reliable for decisions that must happen on
   every turn.** An earlier design deferred the "which tool do I even
   call" decision to lazy skill-discovery — the model skipped the
   knowledge-base tool entirely and hallucinated a file path on the first
   live run. Moving that decision into the always-loaded system prompt
   (`ROUTE_SYSTEM_PROMPT`) fixed it. Lesson: skill/lazy-loading is a good
   fit for guidance that helps when relevant and costs nothing when it
   doesn't apply (procedural depth, edge cases) — a poor fit for rules
   that must apply on every single turn regardless of topic.
4. **A real checkpointer beats a hand-rolled history list.** The first
   version of this app passed prior turns as a hand-rolled,
   tool-call-stripped `history` list into every call, because each call
   built a brand-new agent from scratch. Rebuilt on LangGraph's own
   checkpointer + `thread_id` pattern (build once, invoke many — see
   "Context management" below): a real, live, two-turn conversation ("my
   favorite number is 42" → "what is my favorite number?" → "42")
   round-tripped through LangGraph's own persistence with zero manual
   history reconstruction.
5. **Delegation budgets are advisory, not enforced.** deepagents has no
   framework-level ceiling on subagent/tool-call counts — the
   `max_concurrent_research_units`/`max_researcher_iterations` values used
   here are interpolated into prompt text, which a misbehaving model can
   still exceed. Worth knowing before relying on them as a real guardrail;
   an enforced ceiling would need custom middleware.


## Troubleshooting: web search and benchmark

- The runtime `.env` is **`poc/langgraph_rag/.env`** when viewed from the
  repository root. Copy it from `poc/langgraph_rag/.env.example`.
- `TAVILY_API_KEY` (or `WEB_SEARCH_API_KEY`) is only required for questions
  that the agent routes to live web search. Document-only RAG does not require
  Tavily.
- The LiteLLM/vLLM application backends use LangChain's `ChatOpenAI`, so
  `langchain-openai` is an explicit application dependency.
- After seeding the benchmark documents, run:
  from `poc/langgraph_rag/`.


## Optional primary-source MCP

EvidenceFlow can optionally query live government/official datasets through the Katzilla MCP server. This adds a third evidence branch alongside the internal knowledge base and web research while preserving source URL, retrieval time, quality metadata, and SHA-256 data hash in the UI.

Set `KATZILLA_ENABLED=true` and `KATZILLA_API_KEY=<server-side-key>` to enable it. The integration uses MCP Streamable HTTP; if the connector is unavailable, normal KB/web routing remains available.
