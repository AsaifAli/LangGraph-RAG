"""Lean custom-LangGraph agent — no deepagents. Now the production path
`streamlit_app.py` runs (replacing `unified_poc.py`'s deepagents-based
agent, deleted — see README "Comparison: deepagents/LangGraph vs. current
Agno orchestration" for the full reasoning trail).

Built to answer a specific, measurable question raised while evaluating
deepagents for this app: how much of a deepagents agent's per-turn token
cost comes from the RAG mechanics themselves (retrieval, chunk analysis,
synthesis — all of which live in `rag_pipeline.py` and are reused UNCHANGED
here) versus deepagents' own scaffolding (the always-loaded filesystem-tool
descriptions, `task()`/`write_todos` tool schemas, subagent middleware
instructions)? A live run of the deepagents version against a two-fact
policy lookup measured 38,751 total tokens.

Accuracy is deliberately NOT the variable this design changes: this graph
reuses `rag_pipeline.retrieve_with_self_correction` (hybrid dense+sparse
retrieval, local cross-encoder rerank, grade -> rewrite -> retry
self-correction) and the same citation verifier the deepagents version
used, completely unchanged. Only orchestration overhead is different.

Of deepagents' mechanisms, this graph keeps the two that earned their keep
— parallel per-chunk delegation (here: a plain `StateGraph` fan-out via
`Send`, not the `task()` tool + subagent middleware) and keeping large
chunk text out of the synthesis prompt (each chunk is analyzed in its own
isolated node call, same idea as a chunk-analyst subagent, without a
virtual filesystem in between) — and drops the two that turned out to be
pure overhead for this workload: `write_todos` planning (RAG here is a
fixed-shape workflow — route, retrieve, delegate, synthesize — not
open-ended research that benefits from dynamic replanning) and
skills-loading (already proved unreliable enough in the deepagents
version's own live testing that its routing rule had to move into the
always-loaded prompt anyway, so lazy skill discovery wasn't earning
anything here to begin with).

Multi-turn memory: a `messages` state channel (`langgraph.graph.message.add_messages`,
the same reducer LangGraph's own chatbot-with-memory pattern uses) plus a
checkpointer, exactly like the deepagents version's checkpointer+thread_id
pattern (`docs.langchain.com/oss/python/langgraph/add-memory` — confirmed
directly: `StateGraph.compile(checkpointer=...)` works identically
regardless of conditional edges or `Send`-based fan-out). One disclosed,
real simplification versus the deepagents version: there is no equivalent
of deepagents' automatic `SummarizationMiddleware` here, so a very long
conversation's `messages` list grows unbounded — acceptable at this POC's
scale, not silently pretended equivalent.

Non-serializable state, avoided on purpose: `docs.langchain.com/oss/python/langgraph/add-memory`
does not document storing arbitrary Python objects (like dataclass
instances) in checkpointed state, and flags it as a real gap, not a
confirmed-safe pattern. `RetrievedChunk` dataclass instances are therefore
never put in graph state directly — `retrieve_kb` stores `dataclasses.asdict(chunk)`
plain dicts, and callers reconstruct `RetrievedChunk` objects only
transiently, outside the checkpoint boundary (see `_finalize_outcome`).

Streaming token filtering: with `analyze_chunk` firing in parallel across
every retrieved chunk, a naive `on_chat_model_stream` listener would
interleave route/chunk-analysis/synthesis tokens into garbled output.
Verified directly against the installed `langgraph` source
(`langgraph/pregel/_algo.py`, `"langgraph_node": name` in each event's
metadata — the same field LangGraph's own `stream_mode="messages"` filters
on internally) that every event's `metadata["langgraph_node"]` names the
node currently executing; `astream_langgraph_turn` below filters on it to
stream ONLY the `synthesize` node's own tokens.

`Send`-based map-reduce fan-out (`fanout_chunks` -> `analyze_chunk`) is
LangGraph's own documented pattern for "apply one node to a list of items
in parallel" (`docs.langchain.com/oss/python/langgraph/graph-api`), not
something improvised for this POC.

Hierarchical fan-out for multi-document questions (comparison/"across my
documents"): `analyze_chunk` -> `group_chunk_analyses` (a pure convergence
barrier — a conditional edge attached directly to a Send-fanned node fires
once per parallel instance with only that instance's partial state, not the
merged whole, so an explicit static-edge barrier node sits between them) ->
`fanout_documents`, a SECOND or map-reduce over documents actually
represented, one `summarize_document` worker per document -> `synthesize`.
This is the documented Orchestrator-Worker pattern
(`docs.langchain.com/oss/python/langgraph/workflows-agents`: "the
orchestrator breaks down tasks into subtasks... delegates to workers...
synthesizes worker outputs"), applied twice — once per chunk, once per
document — because comparing raw chunk-level findings from several
documents in one final synthesis call asks the model to do its own
grouping-by-document AND the comparison in the same pass; a clean
per-document rollup first gives the final synthesis something to compare
that's already source-organized. Skipped when only one document is
represented (`fanout_documents` routes straight to `synthesize`) — the
rollup's own LLM call has no comparison to earn its cost against.
Compiled subgraphs (a distinct pattern also considered — see
`docs.langchain.com/oss/python/langgraph/use-subgraphs`) weren't used
here: `Send` can only target a plain node function, not a compiled
subgraph, and a private per-document state schema would have needed one
anyway wrapped in a node function — which is exactly what
`summarize_document` already is, so a separate compiled graph object
would have added indirection without a corresponding benefit.

Run (single-shot, no persisted memory — same shape as research_poc.py/data_analysis_poc.py):
    python -m agents.custom_langgraph_poc --query "..." [--backend gemini-api|huggingface]
        [--document-ids <uuid>,...] [--top-k 5]
"""

from __future__ import annotations

import argparse
import asyncio
import operator
import re
import sys
import uuid
from dataclasses import asdict
from typing import Annotated, Any, Callable, AsyncIterator, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.types import RetryPolicy

from shared.quality import citation_quality, evidence_conflict_candidates, numeric_support, should_abstain_from_kb
from retrieval.rag_pipeline import (
    DEFAULT_BACKEND,
    DEFAULT_MAX_WEB_SEARCH_CALLS,
    DEFAULT_TOP_K,
    PLATFORM_TENANT_ID,
    RetrievedChunk,
    TENANT_SCHEMA,
    build_evidence_registry,
    build_langchain_model,
    content_to_text,
    fetch_all_document_chunks,
    get_web_search_tool,
    load_runtime_config,
    make_token_tracker,
    merge_web_search_query,
    retrieve_with_self_correction,
    summarize_token_usage,
    verifier_finalize,
)

# Node-level retry for transient LLM-API failures — e.g. the Gemini 503
# ("high demand") hit LIVE, twice in a row, earlier in this project, which
# previously required a full manual script re-run to recover from since
# nothing retried it automatically. Verified directly against the installed
# langgraph source (`langgraph.types.default_retry_on`) before adopting: the
# DEFAULT predicate already covers this — it retries anything except a
# denylist of exception types that are almost always programming bugs
# (ValueError/TypeError/etc.), which a provider SDK's own 5xx wrapper
# exception (e.g. `google.genai.errors.ServerError`) isn't, so no custom
# predicate is needed. Matches this codebase's existing
# `_to_thread_with_retry` retry shape (3 attempts) for consistency.
_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0, max_interval=8.0)

# Genuine tool-calling for the routing decision — replaces an earlier
# version that hand-parsed a 3-line ROUTE:/MODE:/QUERY: text reply.
# Matches the actual mechanism in LangGraph's own agentic-RAG tutorial
# (docs.langchain.com/oss/python/langgraph/agentic-rag: "generate a query or
# respond" — the model either calls a bound retriever tool or answers
# directly, decided via real `tool_calls`, not parsed text) — this was the
# one step of that tutorial this graph had reimplemented with a different,
# less robust mechanism instead of extending. Real, not cosmetic,
# improvements over the text-parsed version: no risk of a malformed
# ROUTE:/MODE: reply silently falling through to a wrong default; "needs
# both KB and web" is just two tool calls in one turn instead of a hand-rolled
# BOTH enum value; each tool call carries its OWN query, so a combined
# question no longer forces the same phrasing onto both the KB and web
# searches; and it's consistent with how `research_web` already does its own
# tool-calling loop below, instead of being the one node that worked
# differently.
#
# "No tool call" now means what it means in the tutorial too: the model's
# own reply (not a separate respond_direct node/call) IS the final answer —
# a real efficiency win for greetings, identity questions, and follow-ups
# already answered earlier in the conversation, which no longer cost a
# second LLM call at all.
ROUTE_SYSTEM_PROMPT = """You are a document research assistant.

Documents currently in scope: {document_list}

You have access to the internal knowledge-base tool, and may also have an
optional web-search tool when web search is configured.
- search_knowledge_base(query, mode): search the internal knowledge base of
  uploaded documents/policies.
- search_web(query): search the web for current/external information, but only
  when that optional tool is actually available.

Decide whether this question needs one, both, or neither:
- Broad, unscoped request to summarize or get insights from "the
  documents"/"everything" in general, when MORE THAN ONE document is in
  scope and neither a specific document name nor a topic is given -> do NOT
  call any tool; ask which document or topic to focus on instead. Skip this
  and call search_knowledge_base when: only one document is in scope, OR
  the user explicitly says "all documents"/"everything you have" (a
  complete scope — summarize every document, one section each), OR the
  reference is clearly to a single document already in play — including a
  vague "summarize this/it" with no name given, when one document below is
  marked "(most recently added)": treat THAT one as the target. Never guess
  a DIFFERENT, unnamed document instead — if nothing is marked recent and
  no name is given, that's the ambiguous case above; ask, don't guess.
- Broad, comprehensive comparison request ("compare policy A and B", "how
  do these documents differ") when it's not clear WHICH documents to
  compare (more than two in scope, none named, no "all of them" said) -> do
  NOT call any tool; ask which documents to compare instead of guessing. A
  NARROW comparison naming a specific attribute (e.g. "which policy has the
  higher deductible") doesn't need this gate — call search_knowledge_base
  with mode="search".
- Company document/policy/entity question, OR a vague reference to "the
  document"/"it"/"summarize this" that isn't one of the broad cases above
  -> call search_knowledge_base. Never ask "which document do you mean"
  without calling it first — a search that comes back empty is a normal,
  cheap way to learn nothing relevant is in scope.
- Needs company-specific facts AND external context (e.g. "how does our
  cyber sublimit compare to industry norms?") -> call BOTH tools, and
  phrase each tool's own query for its own part so the final answer can
  attribute each part to its source.
- Not clear whether it's about a specific document or general knowledge ->
  call search_knowledge_base first; a search that finds nothing is a cheap,
  safe way to confirm the web is actually what's needed.
- A follow-up to something already answered this turn or earlier this
  conversation using knowledge-base findings you already have -> answer
  from that directly, don't call search_knowledge_base again for the same
  fact.
- A greeting, or a question about who/what you are -> answer directly, no
  tool call. Never describe yourself as a generic chatbot/LLM or name your
  underlying model or provider — you are this app's document assistant.

When calling search_knowledge_base, set mode:
- "search": the question wants a specific fact, figure, or narrow
  comparison of one attribute (e.g. "what is the cyber sublimit", "which
  policy has the higher deductible") — the most relevant passages from each
  document are enough. Default when unsure.
- "summary": the question wants comprehensive coverage of ONE document, or
  explicitly "everything"/"all documents" (e.g. "summarize this", "what
  topics does it cover", "give me an overview") — needs the WHOLE
  document(s), not just the passages most similar to the question.
- "compare": the question wants a broad, comprehensive comparison across
  TWO OR MORE SPECIFICALLY NAMED documents (e.g. "compare policy A and
  policy B", "how do these two documents differ overall") rather than one
  narrow attribute — needs the WHOLE content of each named document so
  nothing on either side is missed, not just their top-k similar passages.

Every tool query must be a complete, self-contained question — expand
pronouns and vague references ("it", "this", "the document", a follow-up
like "what about X") using the conversation above. If mode is "summary" or
"compare" and a specific document is named or clearly implied, use its name
(from the list above) in the query so it can be identified."""

CHUNK_ANALYSIS_PROMPT = """Question: {query}

Document excerpt [{evidence_id}]:
{content}

Write a 1-3 sentence answer to the question using ONLY this excerpt. Tag
every claim with the excerpt's evidence id, e.g. "The limit is CAD 500,000
[EVID: {evidence_id}]." If this excerpt is not relevant, reply exactly:
"Not relevant. [EVID: {evidence_id}]". Never cite a different evidence id.
Include every specific value the excerpt states that bears on the question —
amounts, limits, dates, rates, IDs, statuses — never name an item while
dropping the figure stated right next to it. Only use values actually
present in this excerpt; never invent, estimate, or infer one that isn't
there."""

# Used for SUMMARY/COMPARE mode instead of CHUNK_ANALYSIS_PROMPT above — a
# real gap found LIVE: CHUNK_ANALYSIS_PROMPT asks each chunk to "answer the
# question," with an explicit escape hatch to reply "Not relevant." For a
# targeted SEARCH question that's exactly right. For "summarize this
# document"/"compare these documents," no SINGLE chunk can ever "answer" a
# whole-document summary or a cross-document comparison alone — comparison
# specifically requires seeing more than one document, which no chunk-level
# call can do — so a model following that prompt literally can plausibly
# (and did, live: an entire 12-chunk document came back with nothing usable)
# mark chunks "Not relevant" that are perfectly good source material, simply
# because they don't themselves answer a meta-level question. This prompt
# extracts unconditionally instead — comparison/synthesis happens later, at
# the document-rollup/synthesize stage, once every chunk's content is
# actually available to reason over together.
CHUNK_EXTRACTION_PROMPT = """Topic: {query}

Document excerpt [{evidence_id}]:
{content}

Write a 1-3 sentence extraction of the key facts in this excerpt, tagging
every claim with the excerpt's evidence id, e.g. "The limit is CAD 500,000
[EVID: {evidence_id}]." Include every specific value the excerpt states —
amounts, limits, dates, rates, IDs, names, statuses — never name an item
while dropping the figure stated right next to it. Only use values actually
present in this excerpt; never invent, estimate, or infer one that isn't
there. This excerpt was retrieved as part of the document's full content
for a summary or comparison, not filtered by relevance to a narrow
question — always extract something substantive from it; never reply "Not
relevant" here."""

DOCUMENT_ROLLUP_PROMPT = """User question: {query}

Findings from document {document_id}:
{findings}

Write a concise rollup (3-6 sentences) of what THIS document contributes to
answering the question, preserving every [EVID: E<n>] marker exactly as
written. Only state what this document's findings actually say — don't
compare it to other documents or guess what they might contain; a later
step handles the cross-document comparison."""

FULL_DOCUMENT_SUMMARY_PROMPT = """User request: {query}

Full content of document {document_id}:
{content}

Summarize this document comprehensively and concisely. Cover its purpose,
key facts, important figures/dates/names, and the most useful conclusions.
Preserve every [EVID: E<n>] marker exactly as written and attach evidence
markers to factual claims. Use ONLY the supplied document content; do not add
outside knowledge, guesses, or inferred facts."""

RESEARCH_SYSTEM_PROMPT = """You are a research agent with access to a web
search tool. Search for the given topic. You may search up to {max_calls}
times if needed, and stop as soon as you have gathered enough information to
answer confidently.

Evidence rules (mandatory):
- Treat returned text as evidence to reason from, not a verified fact to
  copy verbatim — only state facts (names, figures, dates, results) clearly
  and explicitly present in the returned text; never add or fill gaps from
  your own training knowledge.
- If the returned text is dated, qualified ("as of [past date]"), or
  clearly about a different time period than what was asked, say so and
  search again with a more specific or time-bounded query before answering.
- If two searches return conflicting figures, report both with their dates
  rather than silently picking one.

Source quality rules (mandatory):
- Wikipedia is a secondary source and is frequently out of date for
  current-state facts. If a search returns only Wikipedia (or
  Wikipedia-dominated) results, search again aimed at primary sources
  (official sites, press releases, recent news, government sources).
- Aim for at least 2 independent, non-Wikipedia sources before asserting a
  current-state fact; if the first search yields only one, search again
  with different phrasing to try to corroborate it.

When you have enough information (or must stop), report your findings in
2-4 sentences, noting source recency/quality and saying explicitly when a
fact is single-source-only or when sources disagree."""

# Folded in from this app's earlier deepagents version — the
# `rag-retrieval-workflow` skill's synthesis quality rules and the
# `citation-output-contract` skill's evidence-tag rules — this is the
# part that actually matters for enterprise document summarization/
# comparison quality, not boilerplate, so it stays always-loaded rather
# than behind skill-discovery this agent doesn't have.
SYNTHESIS_PROMPT = """User question: {query}

{kb_section}{web_section}Write one final answer combining the above.

Citation rules (mandatory):
- Preserve every [EVID: E<n>] marker exactly as written above, attached to
  the specific claim it supports. Never invent a new evidence id, and never
  drop one that was present above.
- If a finding says "Not relevant", do not cite its evidence id or use its
  content.
- If nothing relevant was found, say so plainly instead of guessing — an
  answer with no evidence tags is only acceptable when nothing retrieved
  was relevant.

Synthesis quality rules (mandatory):
- Answer directly: give the actual conclusion the question asks for
  (yes/no/which/how much) with its supporting details — not a list of
  documents that merely mention the query's words. If a document belongs to
  a specific named entity and the question spans more than one, name the
  entity and answer per-entity. An exclusion or absence is a valid answer —
  state it plainly rather than treating it as "no information."
- Only use a finding that specifically and materially addresses the
  question — a same-word-different-subject match (e.g. "theft of data" when
  asked about "theft of money") or generic boilerplate is not relevant;
  drop it even if a finding included it.
- Include every specific value a finding states that bears on the question —
  amounts, limits, dates, rates, IDs — never name an item while dropping the
  figure stated next to it. Never invent or estimate a value not reported.
- If a finding states a total/subtotal/aggregate figure, use that exact
  figure — never recompute your own sum from component values, even if they
  seem to disagree.
- Don't confuse a price/cost figure with a capacity/limit figure — a
  "premium" is what something costs, not what it covers.
- Exhaustive within relevant scope: for a "what does X cover" or comparison
  question, list every applicable item the findings report in that
  category, not just the first match. Depth follows the request: a specific
  question gets full detail, a broad summary gets headline figures.
- When the same fact appears under more than one document, represent it
  under each document it belongs to — don't collapse or drop a duplicate
  citation just because the value repeats.
- Never introduce an entity not present in the findings above, and never
  reuse an entity from an earlier turn as if it were evidence for this one.
- Don't self-contradict: if the answer already states a value from one
  finding, never also say "not found" for that same value elsewhere.
- If both knowledge-base and web findings are present, attribute each part
  of the answer to its source — never blend them into one unattributed claim.
- Format for scanning: short bold headings by topic/document/entity, with
  bullets (one nested level for sub-details like sub-limits or dates) —
  keep it tight, structure should make the answer clearer, never longer."""


def _add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    from langgraph.graph.message import add_messages

    return add_messages(left, right)


class ChunkAnalysisResult(TypedDict):
    evidence_id: str
    document_id: str
    analysis: str


class DocumentSummary(TypedDict):
    document_id: str
    summary: str


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], _add_messages]
    tenant_schema: str
    platform_tenant_id: str
    document_ids: list[str]
    document_names: dict[str, str]  # document_id -> display name, for SUMMARY-mode targeting
    top_k: int
    needs_kb: bool
    needs_web: bool
    kb_mode: str  # "SEARCH" (top-k similarity), "SUMMARY", or "COMPARE" (whole document)
    contextualized_query: str  # search_knowledge_base's own tool-call query
    web_search_query: str  # search_web's own tool-call query (may differ from the KB one)
    chunks: list[dict]  # asdict(RetrievedChunk) — plain dicts, checkpointer-safe
    chunk_analyses: Annotated[list[ChunkAnalysisResult], operator.add]
    document_summaries: Annotated[list[DocumentSummary], operator.add]
    web_summary: str
    web_query: str
    final_answer: str


@tool(parse_docstring=True)
def search_knowledge_base(query: str, mode: str) -> str:
    """Search the internal knowledge base of uploaded documents and policies.

    Args:
        query: A complete, self-contained question — expand pronouns and
            vague references using conversation context.
        mode: "search" for a specific fact/figure/narrow comparison,
            "summary" for comprehensive coverage of one document (or
            explicitly all documents), "compare" for a broad comparison
            across two or more specifically named documents.
    """
    # Body intentionally never executed: `route` reads this tool's schema
    # via bind_tools and inspects the model's resulting `tool_calls`
    # directly — it never calls `.ainvoke()` on this function. The real
    # retrieval work happens in the `retrieve_kb` node.
    raise NotImplementedError("search_knowledge_base is a schema-only tool — route reads tool_calls, never invokes it")


@tool(parse_docstring=True)
def search_web(query: str) -> str:
    """Search the web for current or external information.

    Args:
        query: The search query.
    """
    # Same as search_knowledge_base above — schema-only, never invoked.
    raise NotImplementedError("search_web is a schema-only tool — route reads tool_calls, never invokes it")


# Broader than this app's actually-ingestible types (txt/md/csv/pdf/docx/
# doc — see app/streamlit_app.py's file_uploader) on purpose: this is a
# "does the query LOOK LIKE it's naming a file" detector, not a
# type-validity check — a user can just as easily type "summarize
# sales.xlsx" for a file that was never uploaded at all, and that should
# still resolve to "not found" rather than falling back to "all", same as
# any other unmatched filename-shaped reference.
_FILENAME_LIKE_RE = re.compile(r"\b[\w\-]+\.(?:txt|pdf|docx?|csv|md|xlsx?)\b", re.IGNORECASE)


def _resolve_full_fetch_targets(query: str, document_ids: list[str], document_names: dict[str, str]) -> list[str]:
    """For MODE=SUMMARY or MODE=COMPARE: which document_id(s) to fetch in
    full. A simple substring match of each known document name against the
    contextualized query — deliberately not NLP/fuzzy matching, just what
    the base case needs: SUMMARY names at most one target, COMPARE names two
    or more, and this matches however many actually appear. Falls back to
    every document currently in scope when there's only one anyway, or when
    no name matches AND the query doesn't look like it named a specific
    file — never silently narrows scope on a genuine "summarize everything"
    request (that case already produced a query without a specific name, so
    it correctly falls through to "all"); the COMPARE routing gate already
    asks for clarification before reaching here if the targets were
    genuinely ambiguous, so a COMPARE call arriving here is expected to have
    resolvable names.

    The one exception to "fall back to all": a query that names an actual
    FILE (e.g. "summarize DON.txt") which doesn't match anything in scope.
    Found live, not a guess: this used to fall through to "every document"
    regardless, so a request for a document that plainly doesn't exist got
    a fabricated-looking answer stitched together from whatever OTHER
    documents happened to be in scope, instead of a plain "no such document"
    — `retrieve_kb`'s caller ends up with empty chunks in that case, and
    `SYNTHESIS_PROMPT`'s own "if nothing relevant was found, say so plainly"
    rule already produces exactly that honest answer once given nothing to
    work with, so returning `[]` here is enough; no new error-message
    plumbing needed. Detected via a plain "looks like a filename"
    regex (`word.ext`), not real NLP — deliberately narrow, so an actually
    vague request (no filename-shaped token at all) still falls through to
    "all" below, unaffected."""
    if len(document_ids) <= 1:
        return document_ids
    query_lower = query.lower()
    matched = [
        doc_id
        for doc_id in document_ids
        if document_names.get(doc_id) and document_names[doc_id].lower() in query_lower
    ]
    if matched:
        return matched
    if _FILENAME_LIKE_RE.search(query):
        return []
    return document_ids


def _normalize_evidence_tags(text: str) -> str:
    """Cheap, robust safety net for real citation-tag slips seen LIVE, ALL
    invisible to `_finalize_outcome`'s strict `\\[EVID:\\s*(E\\d+)\\]`
    extraction regex — not cosmetic typos, but claims that silently drop out
    of verification entirely (not flagged "unverified", just never counted
    as proposed at all). Three variants seen, all handled by one pattern
    (an optional EVID:/EID: prefix, then one or more comma-separated ids):

    1. Truncated prefix: "[EID: E3]" instead of "[EVID: E3]".
    2. Bare grouped tags: "[E1, E2, E3]" — no prefix at all.
    3. Prefixed grouped tags: "[EVID: E2, E3, E5]" — prefix present, but
       multiple ids still grouped in one bracket instead of one tag each.

    All three normalize to separate "[EVID: E<n>]" tags, one per id — the
    verifier expects exactly that shape, not a comma-joined list or a
    truncated prefix. "EID" has no other legitimate meaning in this
    system's output, so correcting it is safe, not papering over real
    ambiguity. Applied at every stage that generates evidence-tagged text
    (chunk analysis, document rollup, synthesis) so an early slip can't
    propagate uncorrected through "preserve exactly as written"
    instructions downstream."""

    def _expand(match: "re.Match[str]") -> str:
        ids = re.findall(r"E\d+", match.group(1))
        return " ".join(f"[EVID: {eid}]" for eid in ids)

    return re.sub(r"\[(?:E(?:VID|ID):\s*)?(E\d+(?:\s*,\s*E\d+)*)\]", _expand, text)


def _build_graph(*, model, web_search_tool):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    # Web search is optional.  If no Tavily/API key is configured, do not
    # expose the web tool to the router at all.  This is important: merely
    # having `search_web` in the tool schema can cause the LLM to select it for
    # an ordinary document question, which would then fail at execution time.
    route_tools = [search_knowledge_base]
    if web_search_tool is not None:
        route_tools.append(search_web)
    bound_route_model = model.bind_tools(route_tools)

    async def route(state: GraphState) -> dict:
        history = state.get("messages") or []
        document_ids = state.get("document_ids") or []
        document_names = state.get("document_names") or {}
        # `document_ids` is [demo fixture, ...uploaded, in upload order] (see
        # streamlit_app.py's `_document_ids_in_scope`) — the LAST entry is
        # reliably the most recently uploaded document whenever there's more
        # than one. Marked explicitly here because the model otherwise has NO
        # way to know a document was just uploaded: the upload confirmation
        # message lives only in the Streamlit UI's own message list, never in
        # this graph's actual conversation history — seen LIVE as a real bug:
        # "summarize this document" right after uploading a second document
        # picked the FIRST-listed (demo) document instead, because nothing
        # told the model the second one was the new one.
        document_list_items = [f'"{document_names.get(d, d)}"' for d in document_ids]
        if len(document_list_items) > 1:
            document_list_items[-1] += " (most recently added)"
        document_list = ", ".join(document_list_items) if document_list_items else "none"
        instruction = ROUTE_SYSTEM_PROMPT.format(document_list=document_list)
        if web_search_tool is None:
            instruction += (
                "\\n\\nWeb search is currently unavailable because it is not configured. "
                "Do not call search_web. For questions about uploaded documents, use "
                "search_knowledge_base; for questions that require live web information, "
                "ask the user to configure web search rather than attempting the unavailable tool."
            )
        response = await bound_route_model.ainvoke([SystemMessage(content=instruction)] + list(history))

        tool_calls = getattr(response, "tool_calls", None) or []
        kb_call = next((c for c in tool_calls if c["name"] == "search_knowledge_base"), None)
        web_call = next((c for c in tool_calls if c["name"] == "search_web"), None)

        if not kb_call and not web_call:
            # No tool call = the model's own reply IS the final answer —
            # greetings, identity questions, clarifying questions from the
            # scope gates above, or a follow-up already answered earlier.
            # Saves a second LLM call versus the old separate respond_direct
            # node for exactly this case.
            text = content_to_text(response.content)
            return {"final_answer": text, "messages": [AIMessage(content=text)]}

        update: dict = {"needs_kb": bool(kb_call), "needs_web": bool(web_call)}
        fallback_query = content_to_text(history[-1].content) if history else ""
        if kb_call:
            args = kb_call.get("args", {})
            update["contextualized_query"] = args.get("query") or fallback_query
            mode = str(args.get("mode", "SEARCH")).strip().upper()
            update["kb_mode"] = mode if mode in ("SEARCH", "SUMMARY", "COMPARE") else "SEARCH"
        if web_call:
            args = web_call.get("args", {})
            web_query = args.get("query") or fallback_query
            update["web_search_query"] = web_query
            update.setdefault("contextualized_query", web_query)
        return update

    def route_branches(state: GraphState) -> list[str]:
        branches = []
        if state.get("needs_kb"):
            branches.append("retrieve_kb")
        if state.get("needs_web"):
            branches.append("research_web")
        return branches or [END]

    async def retrieve_kb(state: GraphState) -> dict:
        if state.get("kb_mode") in ("SUMMARY", "COMPARE"):
            # Whole-document mode: fetch every chunk of the target
            # document(s) instead of a top-k similarity search — see
            # fetch_all_document_chunks's docstring for why top-k retrieval
            # alone can't answer "summarize this"/"compare these documents
            # overall" for documents longer than a top-k window. SUMMARY
            # resolves to one document (or "all"); COMPARE resolves to the
            # two-or-more it was asked to compare — same fetch mechanism,
            # same downstream document_summaries rollup, only the target
            # resolution differs (both handled by _resolve_full_fetch_targets).
            target_ids = _resolve_full_fetch_targets(
                state["contextualized_query"], state["document_ids"], state.get("document_names") or {}
            )
            results = await asyncio.gather(
                *[
                    fetch_all_document_chunks(
                        doc_id,
                        tenant_schema=state["tenant_schema"],
                        platform_tenant_id=state["platform_tenant_id"],
                    )
                    for doc_id in target_ids
                ]
            )
            chunks = [c for group in results for c in group]
            for idx, c in enumerate(chunks, start=1):
                c.evidence_id = f"E{idx}"  # renumber globally across documents
            return {"chunks": [asdict(c) for c in chunks]}

        chunks, _effective_query = await retrieve_with_self_correction(
            state["contextualized_query"],
            model=model,
            tenant_schema=state["tenant_schema"],
            platform_tenant_id=state["platform_tenant_id"],
            document_ids=state["document_ids"],
            top_k=state["top_k"],
        )
        return {"chunks": [asdict(c) for c in chunks]}

    def fanout_chunks(state: GraphState):
        chunks = state.get("chunks") or []
        if not chunks:
            return "synthesize"
        kb_mode = state.get("kb_mode", "SEARCH")

        # For a single-document SUMMARY request, retrieve_kb has already
        # fetched the complete document. Summarize that content directly
        # instead of spending an extra LLM call on per-chunk analysis and then
        # another call on synthesis. This is both cheaper and more reliable.
        if kb_mode == "SUMMARY" and len(state.get("document_ids") or []) == 1:
            return [
                Send(
                    "summarize_document",
                    {
                        "query": state["contextualized_query"],
                        "document_id": state["document_ids"][0],
                        "chunks": chunks,
                    },
                )
            ]

        return [
            Send("analyze_chunk", {"query": state["contextualized_query"], "chunk": c, "kb_mode": kb_mode})
            for c in chunks
        ]


    async def analyze_chunk(payload: dict) -> dict:
        chunk = payload["chunk"]
        # SUMMARY/COMPARE: extract unconditionally (CHUNK_EXTRACTION_PROMPT
        # — see its own comment for why relevance-filtering per chunk is
        # wrong for a whole-document/cross-document question). SEARCH: keep
        # relevance filtering, it's correct for a targeted question.
        template = CHUNK_EXTRACTION_PROMPT if payload.get("kb_mode") in ("SUMMARY", "COMPARE") else CHUNK_ANALYSIS_PROMPT
        prompt = template.format(
            query=payload["query"], evidence_id=chunk["evidence_id"], content=chunk["content"]
        )
        response = await model.ainvoke(prompt)
        return {
            "chunk_analyses": [
                {
                    "evidence_id": chunk["evidence_id"],
                    "document_id": chunk["document_id"],
                    "analysis": _normalize_evidence_tags(content_to_text(response.content)),
                }
            ]
        }

    async def group_chunk_analyses(state: GraphState) -> dict:
        # Pure convergence barrier — exists ONLY so the conditional edge
        # below sees the fully-merged `chunk_analyses` list. A conditional
        # edge attached directly to `analyze_chunk` (which runs as N
        # parallel Send instances) would fire once per instance with only
        # that instance's own partial contribution, not the merged whole —
        # a plain static edge (like this one) is what reliably converges
        # parallel branches in LangGraph before the next node runs.
        return {}

    def fanout_documents(state: GraphState):
        # Orchestrator-worker pattern (docs.langchain.com/oss/python/langgraph/workflows-agents):
        # dynamically spawn one summarization worker per document actually
        # represented in this turn's findings — the count isn't known ahead
        # of time, which is exactly what Send-based fan-out is for. Skipped
        # entirely for the (common) single-document case: an extra rollup
        # call buys nothing when there's only one document to "compare."
        analyses = state.get("chunk_analyses") or []
        if not analyses:
            return "synthesize"
        by_doc: dict[str, list[dict]] = {}
        for a in analyses:
            by_doc.setdefault(a["document_id"], []).append(a)
        if len(by_doc) <= 1:
            return "synthesize"
        return [
            Send(
                "summarize_document",
                {"query": state["contextualized_query"], "document_id": doc_id, "analyses": items},
            )
            for doc_id, items in by_doc.items()
        ]

    async def summarize_document(payload: dict) -> dict:
        if payload.get("chunks"):
            # Single-document SUMMARY: summarize the actual document content
            # returned by fetch_all_document_chunks().
            lines = "\n".join(
                f"[{c['evidence_id']}] {c['content']}" for c in payload["chunks"]
            )
            prompt = FULL_DOCUMENT_SUMMARY_PROMPT.format(
                query=payload["query"], document_id=payload["document_id"], content=lines
            )
        else:
            # Multi-document COMPARE: preserve the existing per-document
            # rollup over chunk analyses before the final comparison.
            lines = "\n".join(f"[{a['evidence_id']}] {a['analysis']}" for a in payload["analyses"])
            prompt = DOCUMENT_ROLLUP_PROMPT.format(
                query=payload["query"], document_id=payload["document_id"], findings=lines
            )
        response = await model.ainvoke(prompt)
        return {
            "document_summaries": [
                {"document_id": payload["document_id"], "summary": _normalize_evidence_tags(content_to_text(response.content))}
            ]
        }


    async def research_web(state: GraphState) -> dict:
        """Run web research without exposing a provider tool to Gemini.

        Gemini's native function-calling responses can contain provider-specific
        thought_signature metadata. Passing those tool-call messages through a
        LiteLLM/OpenAI-compatible proxy can lose that metadata, which causes
        Vertex/Gemini to reject the next request with:
        "Function call is missing a thought_signature".

        The router already decided that this turn needs web search, so there is
        no need to make Gemini perform another tool-selection loop here. Call
        Tavily directly, then give the returned evidence to the normal synthesis
        model as plain text. This preserves web search while keeping the RAG
        pipeline provider-agnostic and avoids leaking provider-specific tool
        protocol details through LiteLLM.
        """
        query = state.get("web_search_query") or state["contextualized_query"]
        effective_query = merge_web_search_query(user_message=query, tool_query=query)
        result = await web_search_tool.search(effective_query)
        return {"web_summary": result, "web_query": query}

    async def synthesize(state: GraphState) -> dict:
        kb_section = ""
        if state.get("document_summaries"):
            # Multi-document case: compare already-clean per-document
            # rollups (see summarize_document) rather than a flat pile of
            # raw chunk findings from every document at once — the model
            # doesn't have to do its own grouping-by-document AND comparison
            # in one pass.
            lines = "\n\n".join(
                f"Document {d['document_id']}:\n{d['summary']}" for d in state["document_summaries"]
            )
            kb_section = f"Knowledge base findings, by document:\n{lines}\n\n"
        elif state.get("chunk_analyses"):
            lines = "\n".join(f"[{c['evidence_id']}] {c['analysis']}" for c in state["chunk_analyses"])
            kb_section = f"Knowledge base findings:\n{lines}\n\n"
        web_section = ""
        if state.get("web_summary"):
            web_section = f"Web research findings:\n{state['web_summary']}\n\n"
        prompt = SYNTHESIS_PROMPT.format(
            query=state["contextualized_query"], kb_section=kb_section, web_section=web_section
        )
        response = await model.ainvoke(prompt)
        text = _normalize_evidence_tags(content_to_text(response.content))
        return {"final_answer": text, "messages": [AIMessage(content=text)]}

    graph = StateGraph(GraphState)
    graph.add_node("route", route, retry_policy=_LLM_RETRY_POLICY)
    graph.add_node("retrieve_kb", retrieve_kb, retry_policy=_LLM_RETRY_POLICY)
    graph.add_node("analyze_chunk", analyze_chunk, retry_policy=_LLM_RETRY_POLICY)
    graph.add_node("group_chunk_analyses", group_chunk_analyses)  # no-op barrier, nothing to retry
    graph.add_node("summarize_document", summarize_document, retry_policy=_LLM_RETRY_POLICY)
    graph.add_node("research_web", research_web, retry_policy=_LLM_RETRY_POLICY)
    graph.add_node("synthesize", synthesize, retry_policy=_LLM_RETRY_POLICY)

    graph.add_edge(START, "route")
    # route can end the turn directly (no tool call = its own reply is the
    # final answer) — no separate respond_direct node/call needed anymore.
    graph.add_conditional_edges("route", route_branches, ["retrieve_kb", "research_web", END])
    graph.add_conditional_edges("retrieve_kb", fanout_chunks, ["analyze_chunk", "summarize_document", "synthesize"])
    graph.add_edge("analyze_chunk", "group_chunk_analyses")
    graph.add_conditional_edges(
        "group_chunk_analyses", fanout_documents, ["summarize_document", "synthesize"]
    )
    graph.add_edge("summarize_document", "synthesize")
    graph.add_edge("research_web", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def build_langgraph_agent(
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    document_ids_provider: Callable[[], list[str]] | None = None,
    document_names_provider: Callable[[], dict[str, str]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    backend: str = DEFAULT_BACKEND,
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    """Construct the agent ONCE — same build-once / invoke-many contract the
    deepagents version used (`document_ids_provider` re-read on every turn,
    not captured once, so a document uploaded mid-conversation is
    immediately in scope; `checkpointer` required for multi-turn memory).

    `document_names_provider`: `{document_id: display_name}`, re-read every
    turn like `document_ids_provider` — used by `route`'s prompt (so the
    model reasons about "the Northbridge policy schedule", not a bare UUID)
    and by `_resolve_summary_targets` (SUMMARY-mode document targeting).
    Defaults to an empty mapping, which degrades gracefully: the route
    prompt falls back to showing raw ids, and SUMMARY-mode target resolution
    falls back to "everything in scope" when no name is available to match
    against — never a hard failure, just less precise disambiguation.
    """
    if document_ids_provider is None:
        document_ids_provider = lambda: []  # noqa: E731
    if document_names_provider is None:
        document_names_provider = lambda: {}  # noqa: E731

    model = build_langchain_model(backend)
    # Web search is an optional capability.  The document-RAG path must remain
    # fully functional when no Tavily/API key is configured.  Do not call
    # get_web_search_tool() in that case because it intentionally raises for a
    # missing key; instead build a KB-only router and expose the sidebar status.
    runtime_config = load_runtime_config()
    web_api_key = (runtime_config.web_search.web_search_api_key or "").strip()
    web_search_tool = get_web_search_tool() if web_api_key else None
    graph = _build_graph(model=model, web_search_tool=web_search_tool)
    compiled = graph.compile(checkpointer=checkpointer)

    return {
        "agent": compiled,
        "web_search_tool": web_search_tool,
        "document_ids_provider": document_ids_provider,
        "document_names_provider": document_names_provider,
        "tenant_schema": tenant_schema,
        "platform_tenant_id": platform_tenant_id,
        "top_k": top_k,
    }


def _turn_input(built: dict[str, Any], query: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "tenant_schema": built["tenant_schema"],
        "platform_tenant_id": built["platform_tenant_id"],
        "document_ids": built["document_ids_provider"](),
        "document_names": built["document_names_provider"](),
        "top_k": built["top_k"],
    }


def _finalize_outcome(state_values: dict, token_tracker, web_search_tool) -> dict[str, Any]:
    """Shared by `run_langgraph_turn`/`astream_langgraph_turn`: turns raw
    graph state into the same outcome shape the deepagents version produced
    (`plan`/`delegations` kept for UI compatibility — `plan` is always `[]`
    since this graph has no `write_todos` equivalent; `delegations` are real,
    not faked, one entry per chunk-analysis call and one for web research)."""
    final_message = state_values.get("final_answer", "")
    chunk_dicts = state_values.get("chunks") or []
    chunks = [RetrievedChunk(**d) for d in chunk_dicts]
    token_usage = summarize_token_usage(token_tracker)

    registry = build_evidence_registry(chunks)
    evidence_ids = sorted(set(re.findall(r"\[EVID:\s*(E\d+)\]", final_message)))
    evidence_refs = [
        {
            "evidence_id": eid,
            "document_id": registry["eid_to_doc"].get(eid, ""),
            "content": registry["eid_to_content"].get(eid, ""),
        }
        for eid in evidence_ids
    ]
    verified = verifier_finalize(evidence_refs, [], registry, mode="verified") if chunks else None
    verified_refs = verified.references if verified else []
    quality = citation_quality(
        proposed_count=len(evidence_refs),
        verified_count=len(verified_refs),
        grounding_status=verified.grounding_status if verified else None,
    )
    quality["numeric_support"] = numeric_support(
        final_message, [r.get("content", "") for r in verified_refs]
    ) if verified_refs else {"numeric_claims_supported": None, "unsupported_values": []}
    quality["evidence_conflicts"] = evidence_conflict_candidates(final_message, verified_refs)

    web_used = bool(state_values.get("web_summary"))
    abstain = should_abstain_from_kb(
        kb_requested=bool(state_values.get("needs_kb")),
        chunk_count=len(chunks),
        verified_count=len(verified_refs),
        web_used=web_used,
    )
    if abstain:
        final_message = (
            "I couldn't find sufficient verified evidence in the available "
            "knowledge base to answer this reliably. I won't invent an answer "
            "from unsupported information."
        )
        quality["quality_label"] = "ABSTAINED"

    delegations = [
        {
            "subagent_type": "chunk-analyst",
            "description": f"Analyze retrieved chunk {c['evidence_id']}",
            "result": c["analysis"],
        }
        for c in (state_values.get("chunk_analyses") or [])
    ]
    delegations.extend(
        {
            "subagent_type": "document-summarizer",
            "description": f"Roll up findings for document {d['document_id']}",
            "result": d["summary"],
        }
        for d in (state_values.get("document_summaries") or [])
    )
    if state_values.get("web_summary"):
        delegations.append(
            {
                "subagent_type": "web-research",
                "description": f"Web search for: {state_values.get('web_query', '')}",
                "result": state_values["web_summary"],
            }
        )

    return {
        "final_answer": final_message,
        "chunks": chunks,
        "used_knowledge_base": bool(chunks),
        "used_web": web_used,
        "grounding_status": verified.grounding_status if verified else None,
        "abstained": abstain,
        "abstention_reason": "NO_VERIFIED_KB_EVIDENCE" if abstain else None,
        "reasons": verified.reasons if verified else {},
        "verified_count": len(verified.references) if verified else 0,
        "proposed_count": len(evidence_refs),
        "quality": quality,
        "token_usage": token_usage,
        "web_sources": web_search_tool.get_captured_sources() if web_search_tool else [],
        "used_web_sources": bool(web_search_tool and web_search_tool.get_captured_sources()),
        "plan": [],
        "delegations": delegations,
    }


async def run_langgraph_turn(built: dict[str, Any], query: str, *, thread_id: str) -> dict[str, Any]:
    """Invoke an already-built agent for one turn. Only the new message is
    sent — with a checkpointer, LangGraph retrieves and prepends prior
    `messages` for that `thread_id` automatically (same contract as the
    deepagents version's `run_unified_turn`)."""
    agent = built["agent"]
    if built.get("web_search_tool") is not None:
        built["web_search_tool"].set_rehydration_context()
    token_tracker = make_token_tracker()
    result = await agent.ainvoke(
        _turn_input(built, query),
        config={"configurable": {"thread_id": thread_id}, "callbacks": [token_tracker]},
    )
    return _finalize_outcome(result, token_tracker, built["web_search_tool"])


async def astream_langgraph_turn(
    built: dict[str, Any], query: str, *, thread_id: str
) -> AsyncIterator[dict[str, Any]]:
    """Streaming variant of `run_langgraph_turn`, matching the deepagents
    version's event shape (`status`/`token`/`final`) so `streamlit_app.py`
    needs no changes to its event-consumption loop. `plan_update`/
    `subagent_start`/`subagent_end` events are not emitted mid-run (this
    graph has no dynamic planning, and its "delegations" — the parallel
    chunk analyses — complete before the token stream begins, unlike
    deepagents' interleaved tool-call events) — the final `delegations` list
    still arrives in the terminal `final` event, so the UI's delegation
    expander still renders, just populated once at the end rather than live.

    Token filtering: `route`, `analyze_chunk` (fired in parallel per chunk),
    and `synthesize` all call the model — without filtering, their
    `on_chat_model_stream` deltas would interleave into garbled text. Every
    event's `metadata["langgraph_node"]` names the node currently executing
    (verified directly against the installed `langgraph` source — see
    module docstring); only `synthesize`'s and `route`'s tokens are streamed
    to the UI — `route`'s call also produces the final answer directly when
    it decides no tool is needed (see `route`'s own docstring comment), and
    its content is empty/near-empty on the tool-calling turns (models
    typically emit tool_calls without accompanying prose), so including it
    doesn't leak anything in the KB/web-routed case.
    """
    agent = built["agent"]
    if built.get("web_search_tool") is not None:
        built["web_search_tool"].set_rehydration_context()
    token_tracker = make_token_tracker()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [token_tracker]}

    _STATUS_LABELS = {
        "retrieve_kb": "🔍 Searching the knowledge base",
        "analyze_chunk": "📖 Analyzing a retrieved chunk",
        "summarize_document": "📑 Rolling up findings per document",
        "research_web": "🌐 Searching the web",
        "synthesize": "✍️ Synthesizing the answer",
    }
    _STREAM_NODES = {"synthesize", "route"}
    seen_status: set[str] = set()

    async for event in agent.astream_events(_turn_input(built, query), config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        node = (event.get("metadata") or {}).get("langgraph_node")
        if kind == "on_chain_start" and name in _STATUS_LABELS and name not in seen_status:
            seen_status.add(name)
            yield {"type": "status", "text": f"{_STATUS_LABELS[name]}..."}
        elif kind == "on_chat_model_stream" and node in _STREAM_NODES:
            delta = content_to_text(getattr(event["data"]["chunk"], "content", None))
            if delta:
                yield {"type": "token", "text": delta}

    state = await agent.aget_state(config)
    outcome = _finalize_outcome(state.values, token_tracker, built["web_search_tool"])
    outcome["type"] = "final"
    yield outcome


async def run_custom_langgraph(
    *,
    query: str,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    document_ids: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    backend: str = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """CLI-friendly single-shot wrapper: build the agent and run exactly one
    turn with no persisted memory — the standalone comparison-script shape
    (matches research_poc.py/data_analysis_poc.py), not the multi-turn chat UI's
    build-once/invoke-many usage (`build_langgraph_agent` + `run_langgraph_turn`/
    `astream_langgraph_turn` above, which is what `streamlit_app.py` uses)."""
    fixed_ids = document_ids or []
    built = build_langgraph_agent(
        tenant_schema=tenant_schema,
        platform_tenant_id=platform_tenant_id,
        document_ids_provider=lambda: fixed_ids,
        top_k=top_k,
        backend=backend,
    )
    return await run_langgraph_turn(built, query, thread_id=uuid.uuid4().hex)


async def _run_cli(args: argparse.Namespace) -> None:
    document_ids = [d for d in (args.document_ids or "").split(",") if d]
    outcome = await run_custom_langgraph(
        query=args.query,
        tenant_schema=args.tenant_schema,
        platform_tenant_id=args.platform_tenant_id,
        document_ids=document_ids,
        top_k=args.top_k,
        backend=args.backend,
    )

    print("\n=== FINAL ANSWER ===\n")
    print(outcome["final_answer"])

    print(f"\nUsed knowledge base this turn: {outcome['used_knowledge_base']}")
    if outcome["used_knowledge_base"]:
        print("\n=== CITATION VERIFICATION (citations/verifier.py) ===\n")
        print(f"grounding_status: {outcome['grounding_status']}")
        print(f"reasons: {outcome['reasons']}")
        print(f"verified references: {outcome['verified_count']} / {outcome['proposed_count']} proposed")

    print(f"\nUsed web search this turn: {outcome['used_web']}")

    usage = outcome["token_usage"]
    print("\n=== TOKEN USAGE ===\n")
    print(f"total: {usage['total_tokens']}  (input: {usage['input_tokens']}, output: {usage['output_tokens']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--tenant-schema", default=TENANT_SCHEMA)
    parser.add_argument("--platform-tenant-id", default=PLATFORM_TENANT_ID)
    parser.add_argument(
        "--document-ids",
        default="",
        help="Comma-separated document UUIDs. Empty means no knowledge-base scope.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--backend", choices=["gemini-api", "huggingface"], default=DEFAULT_BACKEND
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run_cli(args))
    except Exception as exc:  # POC diagnostic path, not production error handling
        print(f"\n[custom_langgraph_poc] run failed: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
