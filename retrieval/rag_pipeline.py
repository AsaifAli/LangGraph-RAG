"""Core retrieval pipeline: config/secrets loading, document ingestion and
chunking, hybrid (dense + sparse) Qdrant retrieval with local cross-encoder
reranking, self-correcting retrieval (grade -> rewrite -> retry), web
search, and chat-model construction. Everything `agents/custom_langgraph_poc.py`
and `app/streamlit_app.py` need to answer a question against the knowledge
base or the web lives here.

Built entirely on LangChain-native primitives (`langchain-qdrant`,
`fastembed`) rather than a heavier agent framework's own vector-db/rerank
wrapper classes — no `agno` (or any other agent-framework) dependency at all.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID


_POC_DIR = Path(__file__).resolve().parent.parent


async def _to_thread_with_retry(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable off the event loop with bounded retries.

    Qdrant/vector-store operations and local model inference are synchronous.
    Keeping them behind one helper prevents them from blocking Streamlit's
    async bridge and gives transient connection/model-loading failures a small
    retry window.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _get_env(key: str, default: str) -> str:
    """Read project config from `.env` / process environment.

    Clean variable names are preferred; legacy ``POC_*`` aliases remain
    accepted so existing local setups keep working. This module never reads
    a parent/repository `.env` file.
    """
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    legacy_key = f"POC_{key}" if not key.startswith("POC_") else key
    return (
        values.get(key)
        or os.environ.get(key)
        or values.get(legacy_key)
        or os.environ.get(legacy_key)
        or default
    )

# --- Tenant/platform namespace for retrieval. Demo seeding is isolated to
# scripts/seed_demo_kb.py; the application itself never injects a demo
# document into a user's scope.
TENANT_SCHEMA = _get_env("TENANT_SCHEMA", "poc_demo")
PLATFORM_TENANT_ID = _get_env("PLATFORM_TENANT_ID", "langgraph_rag")

# --- Chat backend. The default is the local LiteLLM gateway, which keeps
# provider credentials out of the application and supports OpenRouter/Gemini
# fallback routing. Direct Gemini/HF/vLLM backends remain available for local
# experiments.
DEFAULT_BACKEND = _get_env("BACKEND", "litellm")

# --- Model ids per backend. Every one of these needed real live probing
# before being set, not a guess — see README "Backends actually available"
# for the specific 404s/quota errors that ruled out the more obvious model
# names on each backend:
#   - HF: tool-calling confirmed working (proper `tool_calls` response) and
#     ungated (no license-acceptance gate) on the token in .env.
#   - gemini-api: dated model names (gemini-2.5-flash, gemini-2.5-flash-lite)
#     return 404 "no longer available to new users" on this key's project;
#     gemini-2.5-pro has a hard ZERO free-tier quota. Only "-latest" alias
#     names work here, and different aliases are genuinely separate quota
#     buckets (confirmed live).
HF_CHAT_MODEL = _get_env("HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_EMBEDDING_MODEL = _get_env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Sparse leg of hybrid retrieval (see `build_sparse_embedder`) — a BM25-style
# sparse encoder run via `fastembed`. Named-vector identifiers below are
# stable and descriptive rather than tied to any external naming convention.
DEFAULT_SPARSE_MODEL = _get_env("SPARSE_MODEL", "Qdrant/bm25")
DENSE_VECTOR_NAME = _get_env("DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = _get_env("SPARSE_VECTOR_NAME", "sparse")
GEMINI_API_MODEL = _get_env("GEMINI_API_MODEL", "gemini-flash-lite-latest")

# --- Retrieval/chunking/delegation tuning — see chunk_text/chunk_csv_text/
# retrieve_and_rerank/build_langgraph_agent docstrings below for what each one
# actually controls.
DEFAULT_TOP_K = int(_get_env("TOP_K", "5"))
DEFAULT_MAX_CONCURRENT_SUBAGENTS = int(_get_env("MAX_CONCURRENT_SUBAGENTS", "3"))
# research_poc.py only: how many rounds of research delegation before the
# orchestrator must stop, even if sources are still incomplete — a
# different knob from concurrency-per-round above.
DEFAULT_MAX_RESEARCHER_ITERATIONS = int(_get_env("MAX_RESEARCHER_ITERATIONS", "3"))
DEFAULT_CHUNK_SIZE = int(_get_env("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", "150"))
DEFAULT_CSV_OVERLAP_ROWS = int(_get_env("CSV_OVERLAP_ROWS", "1"))
DEFAULT_WEB_SEARCH_MAX_RESULTS = int(_get_env("WEB_SEARCH_MAX_RESULTS", "3"))
# custom_langgraph_poc.py's research_web node: hard cap, enforced in code
# (not just a prompt instruction the model could ignore — the same "advisory
# limits get ignored" lesson DEFAULT_RERANK_MIN_RESULTS/self-correction's
# single-retry cap already apply elsewhere), on how many tavily_search calls
# one research loop can make before being forced to answer with what it has.
DEFAULT_MAX_WEB_SEARCH_CALLS = int(_get_env("MAX_WEB_SEARCH_CALLS", "5"))
# A local ONNX cross-encoder run via `fastembed` — no API key, no external
# call, so it always actually runs (unlike a hosted reranking API, which
# would add a hard dependency on that provider's uptime and quota to every
# retrieval call).
DEFAULT_RERANK_MODEL = _get_env("RERANK_MODEL", "jinaai/jina-reranker-v1-turbo-en")
# Cross-encoder relevance floor below which a candidate is dropped before it
# ever reaches a chunk-analyst subagent call. Tuned for this specific
# reranker model's score distribution (0.0 default is a safe floor — see
# DEFAULT_RERANK_MIN_RESULTS below for what handles the "everything scored
# low" case a raw threshold alone can't).
DEFAULT_RERANK_SCORE_THRESHOLD = float(_get_env("RERANK_SCORE_THRESHOLD", "0.0"))
# Floor on how many candidates `retrieve_and_rerank` returns even when EVERY
# one scores below the threshold above — added after a real, live failure:
# a vague/summarization-style query ("summarize the document") scored every
# real, correctly-scoped chunk negative (a cross-encoder is QA-relevance-
# tuned, not summarization-tuned), so the threshold alone turned a genuinely
# present document into a hard "nothing found." See retrieve_and_rerank's
# fallback comment for the full reasoning.
DEFAULT_RERANK_MIN_RESULTS = int(_get_env("RERANK_MIN_RESULTS", "2"))
# Grade -> rewrite -> re-retrieve self-correction loop (retrieve_with_self_correction
# below), matching LangGraph's own agentic-RAG pattern
# (docs.langchain.com/oss/python/langgraph/agentic-rag) — added to close a gap
# found comparing that pattern against this POC's retrieval directly:
# reranking already filters a bad CANDIDATE POOL (see DEFAULT_RERANK_SCORE_THRESHOLD/
# DEFAULT_RERANK_MIN_RESULTS above), but nothing previously caught a bad QUERY
# and tried again. Costs one extra LLM call per search_knowledge_base call
# (two if a rewrite actually triggers) — real money/quota on the free-tier
# backends this POC runs against (see build_langchain_model's GEMINI_API_MODEL
# comment on how tight that quota already is), so this is an opt-out, not
# unconditionally on.
RAG_SELF_CORRECT_ENABLED = _get_env("RAG_SELF_CORRECT_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
# How many grade -> rewrite -> retry rounds retrieve_with_self_correction can
# take before giving up and surfacing the best result found so far. Open-
# ended in spirit (the model itself decides each round whether another
# rewrite is worth trying, via grade_retrieval_relevance), but genuinely
# unbounded is irresponsible: a query with no real answer in the KB could
# have the grader judge every attempt "not relevant" forever, burning LLM
# calls with no way to stop — same reasoning as DEFAULT_MAX_WEB_SEARCH_CALLS
# for the web-research loop. A real, enforced-in-code ceiling, not a prompt
# instruction the model could ignore.
DEFAULT_MAX_SELF_CORRECT_ROUNDS = int(_get_env("MAX_SELF_CORRECT_ROUNDS", "4"))
# custom_langgraph_poc.py's SUMMARY-mode retrieval (fetch_all_document_chunks
# below): a hard cap, enforced in code, on how many chunks of one document
# get analyzed for a "summarize this whole document" request. Top-k
# similarity retrieval (retrieve_and_rerank) doesn't need this — it already
# caps at top_k by construction — but a full-document fetch has no natural
# ceiling, and analyzing hundreds of chunks in parallel (one analyze_chunk
# Send per chunk) would be a real cost/latency spike for a genuinely long
# document. Not a real hierarchical-batching solution for documents bigger
# than this cap — a disclosed POC-scale limit, not silently pretended away.
DEFAULT_MAX_SUMMARY_CHUNKS = int(_get_env("MAX_SUMMARY_CHUNKS", "40"))

# --- streamlit_app.py's hard backstop on a single turn (see its `_run_async`
# docstring for why this exists at all — no single turn should be able to
# freeze the UI indefinitely regardless of which dependency stalls and why).
DEFAULT_TURN_TIMEOUT_SECONDS = int(_get_env("TURN_TIMEOUT_SECONDS", "180"))

# data_analysis_poc.py only: which bundled fixture file its code-interpreter
# demo loads. A relative path (resolved against poc/langgraph_rag/ by the
# caller), not absolute, so overriding it still stays inside the POC dir
# unless an absolute path is deliberately given.
DEFAULT_FIXTURE_RELATIVE_PATH = _get_env(
    "FIXTURE_PATH", "fixtures/sample_policy_schedule.csv"
)


def get_hf_token() -> str:
    """HF_TOKEN from this app's own `.env` (poc/langgraph_rag/.env), required
    for the "huggingface" chat backend and for the local embedder's
    one-time model download."""
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    token = values.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"HF_TOKEN not found in {_POC_DIR / '.env'}. Add HF_TOKEN=<your token> there."
        )
    return token


def get_gemini_api_key() -> str:
    """GEMINI_API_KEY from this app's own `.env` (poc/langgraph_rag/.env) —
    a direct Google AI Studio API key (`langchain_google_genai.ChatGoogleGenerativeAI`),
    required for the default "gemini-api" chat backend."""
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    key = values.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            f"GEMINI_API_KEY not found in {_POC_DIR / '.env'}. Add GEMINI_API_KEY=<your key> there."
        )
    return key


def load_runtime_config():
    """Load config via `config/loader.py` (reads this app's own `.env`).

    Used for: Qdrant URL/API key, the hybrid-retrieval prefetch multiplier,
    and web-search config.
    """
    from config.loader import load_config

    return load_config()


@dataclass
class RetrievedChunk:
    """One reranked KB chunk, shaped for the deepagents RAG pattern (file-per-chunk)."""

    evidence_id: str
    document_id: str
    content: str
    score: float
    meta_data: dict


_EMBEDDER_CACHE: dict = {}
_SPARSE_EMBEDDER_CACHE: dict = {}


def build_embedder():
    """The embedder used for every collection this POC touches (both seeding
    demo data and querying it): Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`,
    run LOCALLY (`langchain_huggingface.HuggingFaceEmbeddings`, backed by the
    `sentence-transformers`/`torch` packages) rather than via the HF
    Inference API — by explicit request, so embedding calls no longer
    depend on HF Inference credits/availability at all (only the chat
    backend does, and only when `backend="huggingface"`). First call
    downloads and caches the model weights from the HF Hub (a few hundred
    MB, one-time, no HF_TOKEN required — this model is public/ungated);
    every call after that is fully offline.

    `qdrant_collection_name` keys the collection name off `HF_EMBEDDING_MODEL`
    alone, so running locally vs. via a hosted API for the SAME model id is
    expected to produce numerically compatible vectors (same weights, same
    standard mean-pooling + normalization this model ships with) — verified
    directly, not assumed, by comparing a local embedding's dimensionality
    and cosine similarity against a known match. This app only ever reads/
    writes its own clearly-namespaced demo collection (see seed_demo_kb.py).

    Cached per model name in `_EMBEDDER_CACHE` (same pattern
    `_get_rerank_encoder` below already uses for the reranker) — found live
    to matter, not just in principle: every uncached call reconstructs
    `HuggingFaceEmbeddings`, which loads the full `sentence-transformers`
    model from disk into memory again, a multi-second cost every single
    chat-attach upload was paying before this cache existed, on top of the
    chunking/upsert work that's the only part that actually SHOULD scale
    with document size.
    """
    embedder = _EMBEDDER_CACHE.get(HF_EMBEDDING_MODEL)
    if embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
        _EMBEDDER_CACHE[HF_EMBEDDING_MODEL] = embedder
    return embedder


def build_sparse_embedder():
    """Local BM25 sparse encoder (`langchain_qdrant.FastEmbedSparse`, backed
    by `fastembed`'s `Qdrant/bm25` model) — the sparse leg of hybrid
    retrieval. Runs in-process (no external API call, so nothing leaves the
    process and there's no per-call cost). Paired with the dense embedder
    (`build_embedder`) via `RetrievalMode.HYBRID` in
    `ingest_document`/`retrieve_and_rerank`/`seed_demo_kb.py` — Qdrant's own
    Query API fuses both legs server-side (prefetch + RRF), not a
    client-side approximation. Cached per model name, same reasoning as
    `build_embedder` above."""
    sparse = _SPARSE_EMBEDDER_CACHE.get(DEFAULT_SPARSE_MODEL)
    if sparse is None:
        from langchain_qdrant import FastEmbedSparse

        sparse = FastEmbedSparse(model_name=DEFAULT_SPARSE_MODEL)
        _SPARSE_EMBEDDER_CACHE[DEFAULT_SPARSE_MODEL] = sparse
    return sparse


def qdrant_collection_name(tenant_schema: str, platform_tenant_id: str) -> str:
    """Collection naming convention: `{tenant_schema}_{platform_tenant_id}_{embedding_model}_{suffix}`
    — encodes the embedding model into the name so switching models can
    never silently mix incompatible vector spaces in one collection."""
    from shared.constants import AIConstants

    return f"{tenant_schema}_{platform_tenant_id}_{HF_EMBEDDING_MODEL}_{AIConstants.COLLECTION_SUFFIX}".replace(
        "/", "-"
    )


def chunk_text(text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Dynamic, content-adaptive chunker for user-uploaded prose documents —
    LangChain's `RecursiveCharacterTextSplitter` (`langchain_text_splitters`;
    verified directly against the installed source, not assumed from docs:
    default separators `["\\n\\n", "\\n", " ", ""]`), replacing an earlier
    hand-rolled fixed-size splitter (paragraph-pack, then a single hard
    character-window fallback with no intermediate level). Recursively
    tries each separator in turn — paragraph, then line, then word, then
    bare character — falling through to the next only when a unit is still
    larger than `chunk_size`, so a chunk boundary lands on the most natural
    break actually available, rather than always the same fixed strategy
    regardless of content shape. `overlap` (`chunk_overlap`) carries
    trailing context into the next chunk the same recursive way, so a fact
    split across a boundary still has a chance to appear whole somewhere.
    `chunk_overlap > chunk_size` raises a clear `ValueError` from the
    library itself at construction time — no separate clamp needed here
    (verified against the installed splitter: `overlap == chunk_size` is
    accepted, only strictly-greater raises — see `tests/test_chunking.py`).

    A simple, general-purpose chunker — a structure-aware chunking service
    (splitting on document sections/headings rather than raw character
    windows) would produce better boundaries for long structured documents,
    but that's a larger piece of infrastructure than this app needs; this
    is good enough to make uploaded documents queryable.

    Tabular content (`.csv`) is deliberately NOT routed through this
    function — see `chunk_csv_text` below, which stays row-based rather
    than switching to a prose-oriented splitter: recursive paragraph/line/
    word splitting has no concept of "a row," and would cut mid-row or
    separate a value from its header as readily as a fixed-size splitter
    would, losing exactly the self-describing-chunk property
    `chunk_csv_text` exists to guarantee.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def chunk_csv_text(
    text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap_rows: int = DEFAULT_CSV_OVERLAP_ROWS
) -> list[str]:
    """CSV-specific variant of `chunk_text`: plain paragraph-packing (which
    splits on blank lines) doesn't help CSV — there usually aren't any, so
    the whole file becomes one "paragraph" that gets hard-split by character
    window, which can cut mid-row and strips the header off every chunk
    after the first. A bare data row without its header is close to
    meaningless (`POL-10231,1000000,25000` — which number is the sublimit,
    which is the deductible?). This instead packs whole data rows per chunk
    and repeats the header row in every chunk, so each chunk stays
    self-describing on its own regardless of which chunk a downstream reader
    happens to see first.

    `overlap_rows`: carries the last N rows of one chunk into the START of
    the next (after the header), mirroring `chunk_text`'s character overlap
    — added because the original version had NO continuity across a chunk
    boundary at all beyond the shared header: a fact that spans adjacent
    rows (a multi-row record, a subtotal row referencing the row above it)
    could land split across two chunks with nothing carrying it whole into
    either one, the exact failure `chunk_text`'s overlap already guards
    against for free text.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header, rows = lines[0], lines[1:]
    if not rows:
        return [header]

    row_groups: list[list[str]] = []
    current_rows: list[str] = []
    current_len = len(header)
    for row in rows:
        if current_rows and current_len + len(row) + 1 > chunk_size:
            row_groups.append(current_rows)
            current_rows, current_len = [], len(header)
        current_rows.append(row)
        current_len += len(row) + 1
    if current_rows:
        row_groups.append(current_rows)

    chunks: list[str] = []
    for i, group in enumerate(row_groups):
        carried = row_groups[i - 1][-overlap_rows:] if i > 0 and overlap_rows > 0 else []
        chunks.append("\n".join([header, *carried, *group]))
    return chunks


def extract_text(content: bytes, source_name: str) -> str:
    """Dispatch to `shared/text_extractors.py::DocumentTextExtractor` by file
    extension (pdfplumber for PDF, python-docx for DOCX including tables,
    a LibreOffice-headless fallback for legacy binary `.doc`). `.txt`/`.md`/
    `.csv` also go through it (`txt_bytes_to_text`), which falls back to
    latin-1 for non-UTF-8 text instead of raising a bare decode error.

    Legacy `.doc` note, disclosed rather than silently broken: the
    LibreOffice fallback (`doc_bytes_to_text_via_libreoffice`) needs a
    `LIBREOFFICE` env var pointing at a real binary; without it, a `.doc`
    upload raises `StorageError(LIBREOFFICE_NOT_FOUND)`. `.docx` does not
    depend on this at all (python-docx handles it directly) and works
    regardless.
    """
    from shared.text_extractors import DocumentTextExtractor

    extractor = DocumentTextExtractor()
    suffix = source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""
    if suffix == "pdf":
        return extractor.pdf_bytes_to_text(content)
    if suffix == "docx":
        return extractor.docx_bytes_to_text(content)
    if suffix == "doc":
        return extractor.doc_bytes_to_text(content)
    return extractor.txt_bytes_to_text(content)


async def ingest_document(
    content: bytes,
    *,
    source_name: str,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    document_id: str | None = None,
) -> dict:
    """Extract text, chunk, embed, and upsert a user-uploaded document into
    the POC's demo Qdrant collection — additive (never `force_recreate`,
    unlike `seed_demo_kb.py`'s fixed demo data), so uploads accumulate
    rather than wiping prior ones. Returns
    `{"document_id": ..., "chunk_count": ...}`; pass the returned
    `document_id` in `document_ids=` on `retrieve_and_rerank` / `run_rag` /
    `run_custom_langgraph` to make the upload queryable.

    `content` is always raw bytes (not pre-decoded text) — `extract_text`
    handles binary formats (PDF/DOCX/DOC) as well as plain text, dispatched
    by `source_name`'s extension.

    Writes BOTH a dense and a sparse (BM25) vector per chunk
    (`RetrievalMode.HYBRID`) — `retrieve_and_rerank` queries both legs via
    Qdrant's native Query API prefetch+RRF (see that function's docstring).
    `force_recreate=False` here means an EXISTING dense-only collection
    (from before hybrid support was added) will NOT be silently upgraded —
    see `seed_demo_kb.py`, which does force-recreate, for how the demo
    collection actually gets its hybrid schema.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import models

    from shared.constants import AIReferenceKeys

    if document_id is None:
        document_id = str(uuid.uuid4())

    text = extract_text(content, source_name)
    config = load_runtime_config()
    collection_name = qdrant_collection_name(tenant_schema, platform_tenant_id)
    chunks = chunk_csv_text(text) if source_name.lower().endswith(".csv") else chunk_text(text)
    if not chunks:
        return {"document_id": document_id, "chunk_count": 0}

    metadatas = [
        {
            "document_id": document_id,
            "tenant_schema": f"{tenant_schema}_{platform_tenant_id}",
            "section_path": f"{source_name} > chunk {i}",
        }
        for i in range(1, len(chunks) + 1)
    ]

    await _to_thread_with_retry(
        QdrantVectorStore.from_texts,
        texts=chunks,
        embedding=build_embedder(),
        sparse_embedding=build_sparse_embedder(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        metadatas=metadatas,
        collection_name=collection_name,
        url=config.vector_db.qdrant_url,
        api_key=config.vector_db.qdrant_api_key,
        distance=models.Distance.COSINE,
        content_payload_key=AIReferenceKeys.CONTENT,
        metadata_payload_key=AIReferenceKeys.META_DATA,
        force_recreate=False,  # append — reuses the collection if it already exists
    )
    return {"document_id": document_id, "chunk_count": len(chunks)}


# Process-wide encoder cache: each fastembed cross-encoder model loads once
# (a real, if small, ONNX model load) and is reused for every subsequent
# call in this process, not reloaded per query.
_RERANK_ENCODER_CACHE: dict = {}


def _get_rerank_encoder(model_name: str):
    encoder = _RERANK_ENCODER_CACHE.get(model_name)
    if encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        encoder = TextCrossEncoder(model_name=model_name)
        _RERANK_ENCODER_CACHE[model_name] = encoder
    return encoder


def _local_cross_encoder_rerank(
    query: str, documents: list, *, top_n: int, model_name: str = DEFAULT_RERANK_MODEL
) -> list[tuple[Any, float]]:
    """Rerank a candidate pool with a local cross-encoder model
    (`jinaai/jina-reranker-v1-turbo-en` via `fastembed`'s `TextCrossEncoder`).
    Runs fully in-process (ONNX) — no API key, no external call, so it
    always actually runs regardless of which external services are
    reachable. Returns `(document, score)` pairs, sorted descending by
    score, capped at `top_n`."""
    encoder = _get_rerank_encoder(model_name)
    contents = [doc.page_content or "" for doc in documents]
    scores = list(encoder.rerank(query, contents))
    ranked = sorted(zip(documents, scores), key=lambda pair: -pair[1])
    return ranked[:top_n] if top_n > 0 else ranked


async def retrieve_and_rerank(
    query: str,
    *,
    tenant_schema: str,
    platform_tenant_id: str,
    document_ids: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve-then-rerank, built entirely on LangChain-native primitives
    (no agent-framework vector-db wrapper class involved):

      1. Native hybrid search via `langchain_qdrant.QdrantVectorStore`
         (`RetrievalMode.HYBRID`) — dense (`build_embedder`) and sparse/BM25
         (`build_sparse_embedder`, `Qdrant/bm25` via `fastembed`) legs run
         as a SINGLE Qdrant Query API call (`prefetch=[dense, sparse]`,
         `query=FusionQuery(fusion=Fusion.RRF)`, confirmed directly from the
         installed `langchain_qdrant` source, not assumed) — genuine
         server-side RRF fusion between two independently-run legs, not a
         client-side approximation. Uses `build_kb_filter_expr` for
         tenant/document scoping (a plain `qdrant_client.models.Filter`).
      2. Local cross-encoder rerank (`_local_cross_encoder_rerank`,
         `jinaai/jina-reranker-v1-turbo-en` via `fastembed`) — runs
         in-process, no API key, no external call, so it always actually
         executes regardless of environment.

    A single collection with named dense+sparse vectors from the start
    (created directly with hybrid schema, no dense-only-to-hybrid migration
    path needed since this is a fresh collection).
    """
    import asyncio

    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient

    from shared.entities import SanitizedQuery, SecureAgentRequest
    from retrieval.kb_filter_builder import build_kb_filter_expr

    config = load_runtime_config()

    request = SecureAgentRequest(
        query=SanitizedQuery(content=query),
        correlation_id=str(uuid.uuid4()),
        platform_tenant_id=platform_tenant_id,
        tenant_schema=tenant_schema,
        document_ids=[UUID(d) for d in document_ids] if document_ids else None,
    )
    filter_expr = build_kb_filter_expr(request)
    collection_name = qdrant_collection_name(tenant_schema, platform_tenant_id)

    from shared.constants import AIReferenceKeys

    client = QdrantClient(url=config.vector_db.qdrant_url, api_key=config.vector_db.qdrant_api_key)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=build_embedder(),
        sparse_embedding=build_sparse_embedder(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        # Explicit payload key names (AIReferenceKeys.CONTENT / .META_DATA),
        # NOT langchain-qdrant's own defaults ("page_content"/"metadata") —
        # required so the filter below, built by build_kb_filter_expr
        # (which filters on the `meta_data.tenant_schema` /
        # `meta_data.document_id` payload paths,
        # VectorDBConstants.META_TENANT_SCHEMA_KEY/META_DOCUMENT_ID_KEY),
        # actually matches points in this collection. seed_demo_kb.py writes
        # with the same keys for exactly this reason.
        content_payload_key=AIReferenceKeys.CONTENT,
        metadata_payload_key=AIReferenceKeys.META_DATA,
    )

    rerank_cfg = config.rerank
    prefetch_limit = max(top_k, top_k * rerank_cfg.hybrid_prefetch_multiplier)

    if len(document_ids) > 1:
        # Multi-document scope (a comparison/"across my documents" question)
        # — fetch candidates PER document (parallel), not one shared query
        # across all of them. A single shared top-k ranking has no fairness
        # guarantee: nothing stops one document's chunks from winning every
        # seat and the other document silently contributing zero candidates
        # to the rerank stage, purely because its content scored lower on
        # THIS query — not because it lacks the answer. Checked live, not
        # just theorized: a 2-document comparison test happened to retrieve
        # the right chunk from both sides this time (the corpus was small
        # enough that everything fit), but nothing in a single shared query
        # would have caught it if it hadn't — this replaces luck with a
        # guarantee. Each document's own filter is built the same way
        # (`build_kb_filter_expr`) as the single-document path below, just
        # scoped to one id per call instead of a MatchAny over all of them.
        per_doc_filters = []
        for doc_id in document_ids:
            doc_request = SecureAgentRequest(
                query=SanitizedQuery(content=query),
                correlation_id=str(uuid.uuid4()),
                platform_tenant_id=platform_tenant_id,
                tenant_schema=tenant_schema,
                document_ids=[UUID(doc_id)],
            )
            per_doc_filters.append((doc_id, build_kb_filter_expr(doc_request)))

        per_doc_results = await asyncio.gather(
            *[
                _to_thread_with_retry(
                    vector_store.similarity_search_with_score, query, k=prefetch_limit, filter=f
                )
                for _doc_id, f in per_doc_filters
            ]
        )
        documents = [doc for results in per_doc_results for doc, _score in results]
    else:
        # Single-document (or no-scope) query — one shared query is already
        # fair by construction, no per-document fan-out needed.
        results = await _to_thread_with_retry(
            vector_store.similarity_search_with_score,
            query,
            k=prefetch_limit,
            filter=filter_expr,
        )
        documents = [doc for doc, _score in results]

    chunks: list[RetrievedChunk] = []
    if documents:
        # Local cross-encoder rerank ALWAYS runs now (no API key gate) —
        # see docstring for why this replaced the Cohere path. Reranks the
        # COMBINED pool (all documents together) ONCE — fairness comes from
        # every document getting a fair SHOT at the pool above, not from
        # reranking each document in isolation, which would lose cross-
        # document relevance ordering entirely. Off the main event loop via
        # _to_thread_with_retry since ONNX inference is synchronous/CPU-bound.
        reranked = await _to_thread_with_retry(
            _local_cross_encoder_rerank, query, documents, top_n=max(top_k, len(documents))
        )
        scored = [(doc, float(score)) for doc, score in reranked]
        above = [pair for pair in scored if pair[1] > DEFAULT_RERANK_SCORE_THRESHOLD]
        # Overall seat count. Computed BEFORE selection, not applied as a slice
        # after it — see the multi-document branch below for why that ordering
        # is the whole point.
        cap = max(top_k, len(document_ids) * DEFAULT_RERANK_MIN_RESULTS)
        # Fallback found LIVE, not theorized: "summarize the document" against
        # real CSV row chunks scored every candidate NEGATIVE (-3.18 to
        # -3.30) — a cross-encoder is tuned for QA-relevance matching a
        # specific question to a specific passage, not "is this generically
        # summarizable," so a vague/summarization-style query can legitimately
        # score every real, on-topic chunk below the floor. Applying the
        # threshold here previously turned a real, present, correctly-scoped
        # document into a hard "nothing found" — the retrieval layer alone
        # deciding "irrelevant" and never giving the model a chance to see
        # ANY candidate, exactly the failure the "search first, ask
        # clarifying questions only after" routing rule was trying to avoid.
        # So: never let a genuinely non-empty candidate pool starve to zero.
        # Both branches below honour that — per document when more than one is
        # in scope, overall when there's just one.
        if len(document_ids) > 1:
            # RESERVE each document's seats first, then fill the remainder by
            # score — rather than score-ranking everything and slicing to `cap`
            # at the end. The slice-last version silently defeated its own
            # guarantee: with 2 documents and top_k=5, a document whose chunks
            # all scored below the other's got appended after the cap and then
            # cut straight back off, so "compare A and B" really did become
            # "describe A" — the exact failure this branch exists to prevent.
            # Verified with the reranker/Qdrant mocked: doc B was absent from
            # the result both when its chunks scored above the threshold and
            # when they scored below it.
            reserved: list[tuple[Any, float]] = []
            for doc_id in document_ids:
                own_above = [p for p in above if str(p[0].metadata.get("document_id")) == doc_id]
                # Fall back to the document's best candidates regardless of
                # score when none clear the floor — same reasoning as the
                # single-document fallback below (a cross-encoder can score
                # every genuinely on-topic chunk negative for a vague or
                # summarization-style query).
                own_any = [p for p in scored if str(p[0].metadata.get("document_id")) == doc_id]
                reserved.extend((own_above or own_any)[:DEFAULT_RERANK_MIN_RESULTS])
            taken = {id(doc) for doc, _ in reserved}
            filler = [pair for pair in above if id(pair[0]) not in taken]
            survivors = reserved + filler[: max(0, cap - len(reserved))]
            # Reserved seats were collected per document, so restore global
            # score order for the caller (evidence ids are assigned in list
            # order, and the best-supported chunk should still be E1).
            survivors.sort(key=lambda pair: -pair[1])
        else:
            survivors = above[:cap]
            if not survivors and scored:
                survivors = scored[:DEFAULT_RERANK_MIN_RESULTS]
        for idx, (doc, score) in enumerate(survivors, start=1):
            chunks.append(
                RetrievedChunk(
                    evidence_id=f"E{idx}",
                    document_id=str(doc.metadata.get("document_id") or "unknown"),
                    content=doc.page_content or "",
                    score=score,
                    meta_data=doc.metadata,
                )
            )
    return chunks


async def grade_retrieval_relevance(model, query: str, chunks: list[RetrievedChunk]) -> bool:
    """One token-cheap LLM call judging whether a retrieved batch is worth
    answering from at all — the "grade_documents" half of LangGraph's
    agentic-RAG loop (see RAG_SELF_CORRECT_ENABLED above), collapsed into a
    SINGLE combined judgment over a short preview of the top 3 chunks rather
    than one grading call per chunk the way the docs example does it — a
    call per chunk would multiply an already-scarce free-tier LLM budget for
    little extra signal beyond one combined yes/no. Called once per round by
    `retrieve_with_self_correction`'s loop, on whatever query/chunks that
    round just retrieved.

    Returns False immediately for an empty batch — nothing to grade, and an
    empty result always deserves a rewrite attempt regardless of what a
    grading call would say.
    """
    if not chunks:
        return False
    preview = "\n\n".join(f"[{c.evidence_id}] {c.content[:200]}" for c in chunks[:3])
    prompt = (
        f"Question: {query}\n\nRetrieved passages:\n{preview}\n\n"
        "Could these passages help answer the question, even partially? "
        "Reply with exactly one word: yes or no."
    )
    response = await model.ainvoke(prompt)
    return content_to_text(response.content).strip().lower().startswith("y")


async def rewrite_search_query(model, query: str, *, prior_attempts: list[str] | None = None) -> str:
    """Second half of the self-correction loop: one LLM call reformulating a
    query that graded NOT-relevant (or came back empty) into a clearer, more
    specific search query. `retrieve_with_self_correction` below calls this
    once per round, up to `DEFAULT_MAX_SELF_CORRECT_ROUNDS` — a real,
    enforced-in-code cap, not a prompt instruction the model could ignore
    (deepagents/LangGraph give no framework-enforced iteration cap of their
    own; see the architecture write-up's "known limitations").

    `prior_attempts`: every query already tried this call (original plus
    every earlier rewrite), told to the model explicitly so round 3 doesn't
    just re-propose round 2's already-failed rewrite — without this, a
    multi-round loop has no memory of its own history and can cycle between
    the same one or two phrasings instead of genuinely exploring new ones."""
    history_section = ""
    if prior_attempts:
        tried = "\n".join(f'- "{q}"' for q in prior_attempts)
        history_section = f"\n\nAlready tried, also unsuccessful — do not repeat any of these:\n{tried}"
    prompt = (
        f'This search against a document knowledge base returned nothing useful:\n"{query}"'
        f"{history_section}\n\n"
        "Rewrite it as a single clearer, more specific search query more likely to match "
        "relevant document text. Reply with ONLY the rewritten query text, nothing else."
    )
    response = await model.ainvoke(prompt)
    rewritten = content_to_text(response.content).strip().strip('"')
    return rewritten or query


async def retrieve_with_self_correction(
    query: str,
    *,
    model,
    tenant_schema: str,
    platform_tenant_id: str,
    document_ids: list[str],
    top_k: int = DEFAULT_TOP_K,
    max_rounds: int = DEFAULT_MAX_SELF_CORRECT_ROUNDS,
) -> tuple[list[RetrievedChunk], str]:
    """`retrieve_and_rerank`, wrapped with the grade -> rewrite -> re-retrieve
    self-correction loop from LangGraph's agentic-RAG pattern
    (docs.langchain.com/oss/python/langgraph/agentic-rag). Reranking already
    filters a bad CANDIDATE POOL; this catches a bad QUERY and retries it —
    open-ended in spirit (the model itself decides each round, via
    `grade_retrieval_relevance`, whether the result is good enough to stop),
    capped at `max_rounds` in code so a query with no real answer in the KB
    can't have the grader say "not relevant" forever (see
    `DEFAULT_MAX_SELF_CORRECT_ROUNDS`'s comment) and can't loop on a rewrite
    it's already tried (`rewrite_search_query`'s `prior_attempts`).

    Returns `(chunks, effective_query)` — `effective_query` is whichever
    query (original or a later rewrite) actually produced the returned
    chunks, so callers (search_knowledge_base) can tell the model what was
    actually searched for.

    Set POC_RAG_SELF_CORRECT_ENABLED=false to skip grading/rewrite entirely
    and behave exactly like a bare `retrieve_and_rerank` call — see
    RAG_SELF_CORRECT_ENABLED's comment for why this is opt-out, not free.
    """

    async def _search(q: str) -> list[RetrievedChunk]:
        return await retrieve_and_rerank(
            q,
            tenant_schema=tenant_schema,
            platform_tenant_id=platform_tenant_id,
            document_ids=document_ids,
            top_k=top_k,
        )

    current_query = query
    chunks = await _search(current_query)
    if not RAG_SELF_CORRECT_ENABLED:
        return chunks, current_query

    best_chunks, best_query = chunks, current_query
    tried = [query]
    for _round in range(max_rounds):
        if await grade_retrieval_relevance(model, current_query, chunks):
            return chunks, current_query

        rewritten_query = await rewrite_search_query(model, current_query, prior_attempts=tried)
        if rewritten_query.strip().lower() in {q.strip().lower() for q in tried}:
            break  # model has nothing genuinely new to try — stop rather than loop on a repeat
        tried.append(rewritten_query)

        chunks = await _search(rewritten_query)
        current_query = rewritten_query
        if chunks:
            best_chunks, best_query = chunks, current_query

    return best_chunks, best_query  # every round exhausted or nothing new to try — best found so far


async def fetch_all_document_chunks(
    document_id: str,
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    max_chunks: int = DEFAULT_MAX_SUMMARY_CHUNKS,
) -> list[RetrievedChunk]:
    """Fetch EVERY chunk of one document — a plain Qdrant payload scroll, NOT
    a similarity search — for genuine "summarize this whole document"/"what
    topics does it cover" requests, where `retrieve_and_rerank`'s top-k
    similarity search would silently only see the handful of chunks most
    similar to the query embedding and miss the rest of a long document.
    Real, structural gap this closes: top-k retrieval is shaped for "find
    the passages relevant to this question," not "read the whole document,"
    and no amount of prompt tuning on the synthesis side fixes a retrieval
    step that never fetched most of the document in the first place.

    No reranking, no relevance score (`score=0.0` on every chunk) — there is
    no query to rank against, only "give me everything." Reuses the exact
    same tenant/document filter (`build_kb_filter_expr`) `retrieve_and_rerank`/
    `delete_document` already use, so "what counts as this document's
    points" stays defined identically everywhere.

    Capped at `max_chunks` (`DEFAULT_MAX_SUMMARY_CHUNKS`), truncating rather
    than fetching an unbounded number of chunks for a very long document —
    disclosed POC-scale limit, not a real hierarchical-batching solution
    for documents bigger than the cap.
    """
    from qdrant_client import QdrantClient

    from shared.entities import SanitizedQuery, SecureAgentRequest
    from shared.constants import AIReferenceKeys
    from retrieval.kb_filter_builder import build_kb_filter_expr

    config = load_runtime_config()
    request = SecureAgentRequest(
        query=SanitizedQuery(content="[fetch all chunks]"),
        correlation_id=str(uuid.uuid4()),
        platform_tenant_id=platform_tenant_id,
        tenant_schema=tenant_schema,
        document_ids=[UUID(document_id)],
    )
    filter_expr = build_kb_filter_expr(request)
    collection_name = qdrant_collection_name(tenant_schema, platform_tenant_id)
    client = QdrantClient(url=config.vector_db.qdrant_url, api_key=config.vector_db.qdrant_api_key)

    chunks: list[RetrievedChunk] = []
    offset = None
    while len(chunks) < max_chunks:
        points, offset = await _to_thread_with_retry(
            client.scroll,
            collection_name=collection_name,
            scroll_filter=filter_expr,
            limit=min(100, max_chunks - len(chunks)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            meta = payload.get(AIReferenceKeys.META_DATA, {}) or {}
            chunks.append(
                RetrievedChunk(
                    evidence_id=f"E{len(chunks) + 1}",
                    document_id=str(meta.get("document_id") or document_id),
                    content=payload.get(AIReferenceKeys.CONTENT, "") or "",
                    score=0.0,
                    meta_data=meta,
                )
            )
            if len(chunks) >= max_chunks:
                break
        if offset is None or not points:
            break
    return chunks


async def delete_document(
    document_id: str,
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
) -> int:
    """Delete every chunk of one document from the demo Qdrant collection —
    a real deletion, not just removal from a local "in scope" list. Reuses
    the exact same `build_kb_filter_expr` already used by `retrieve_and_rerank`
    above to build the delete filter, so "what counts as this document's
    points" is defined
    identically for both retrieval and deletion — no separate, potentially
    drifting filter logic. `QdrantClient.delete(points_selector=<Filter>)`
    deletes every point matching the filter in one call; no need to look up
    point ids first.

    Returns the collection's point count immediately after deletion (a
    cheap, real way to confirm points actually disappeared, not just that
    the API call didn't raise).
    """
    from qdrant_client import QdrantClient

    from shared.entities import SanitizedQuery, SecureAgentRequest
    from retrieval.kb_filter_builder import build_kb_filter_expr

    config = load_runtime_config()
    request = SecureAgentRequest(
        # SanitizedQuery requires non-empty content; the query text itself is
        # irrelevant here — build_kb_filter_expr only reads tenant/document_ids.
        query=SanitizedQuery(content="[delete document request]"),
        correlation_id=str(uuid.uuid4()),
        platform_tenant_id=platform_tenant_id,
        tenant_schema=tenant_schema,
        document_ids=[UUID(document_id)],
    )
    filter_expr = build_kb_filter_expr(request)
    collection_name = qdrant_collection_name(tenant_schema, platform_tenant_id)

    client = QdrantClient(url=config.vector_db.qdrant_url, api_key=config.vector_db.qdrant_api_key)
    await _to_thread_with_retry(client.delete, collection_name=collection_name, points_selector=filter_expr)
    count_result = await _to_thread_with_retry(client.count, collection_name=collection_name, exact=True)
    return count_result.count


def get_web_search_tool(*, max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS):
    """Build a Tavily-backed `WebSearchTool` (`tools/web_search_tool.py`).
    Constructed standalone here since this app has no live chat session/db
    pool; with `db_pool=None` and `is_temp=False`, its query-rehydration hook
    is a pass-through regardless."""
    from tools.web_search_tool import WebSearchTool

    config = load_runtime_config()
    ws_cfg = config.web_search
    api_key = (ws_cfg.web_search_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "Web search is not configured. Add TAVILY_API_KEY or WEB_SEARCH_API_KEY to the project .env to enable it."
        )
    return WebSearchTool(api_key=api_key, max_results=max_results or ws_cfg.web_search_max_results)


def make_token_tracker():
    """A LangChain `UsageMetadataCallbackHandler` (langchain-core, not
    something this POC wrote) that accumulates `AIMessage.usage_metadata`
    across every chat-model call in a run — including every subagent's own
    calls, since they all execute inside the same LangGraph run the callback
    is attached to. Pass the returned handler via
    `agent.ainvoke(..., config={"callbacks": [handler]})`, then read
    `summarize_token_usage(handler)` after the run.

    Caveat, not hidden: accumulation keys on `response_metadata["model_name"]`
    being set on each `AIMessage`, which is populated by each LangChain
    provider integration, not by this POC — if a given backend's integration
    doesn't set it, that backend's calls won't be counted. Not verified
    end-to-end in this sandbox (see README "What was actually tested" for
    why: all three backends fail before completing a turn here).
    """
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    return UsageMetadataCallbackHandler()


def summarize_token_usage(tracker) -> dict:
    """Collapse the tracker's per-model breakdown into a single total, plus
    the per-model breakdown for anyone who wants it."""
    input_tokens = sum(v.get("input_tokens", 0) for v in tracker.usage_metadata.values())
    output_tokens = sum(v.get("output_tokens", 0) for v in tracker.usage_metadata.values())
    total_tokens = sum(v.get("total_tokens", 0) for v in tracker.usage_metadata.values())
    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "by_model": dict(tracker.usage_metadata),
    }


def merge_web_search_query(*, user_message: str, tool_query: str) -> str:
    """Deterministic query-merge step — no LLM rewrite involved. Re-exported
    here from `shared/query_merge.py` for convenience (see that file's
    docstring for the merge logic itself)."""
    from shared.query_merge import merge_web_search_query as _merge

    return _merge(user_message=user_message, tool_query=tool_query)


# --- citation verification (citations/verifier.py) ---
# Pure/stateless, zero I/O, zero agent-framework dependency — safe to
# import directly. Only `finalize` is actually called anywhere in this app
# (custom_langgraph_poc.py) — verifier.py's other public names (FinalizeResult,
# GroundingStatus, ReasonCode, VerifiedRef, verify_reference) were
# re-exported here too but never imported by anything, dead re-exports
# removed rather than kept "just in case."
from citations.verifier import finalize as verifier_finalize  # noqa: E402


def build_langchain_model(backend: str):
    """Build the chat model used by the LangGraph runtime.

    Backends:
      - "litellm": the portfolio OpenAI-compatible gateway.
      - "huggingface": hosted Hugging Face inference.
      - "gemini-api": Google AI Studio.
      - "vllm": a local OpenAI-compatible vLLM server.
    """
    if backend == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=HF_CHAT_MODEL,
            task="text-generation",
            huggingfacehub_api_token=get_hf_token(),
        )
        return ChatHuggingFace(llm=endpoint)

    if backend == "gemini-api":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_API_MODEL,
            google_api_key=get_gemini_api_key(),
        )

    if backend in {"openrouter", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        api_key = _get_env("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the openrouter backend.")
        return ChatOpenAI(
            model=_get_env("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=_get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            temperature=0,
        )

    if backend == "litellm":
        # LiteLLM is the application's gateway; provider credentials belong
        # to the LiteLLM container, not to this client.  Fail early with a
        # useful configuration error instead of allowing an empty/invalid
        # upstream key to turn into a long OpenRouter/Gemini fallback trace.
        from langchain_openai import ChatOpenAI

        from dotenv import dotenv_values

        values = dotenv_values(_POC_DIR / ".env")
        openrouter_key = (
            values.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        ).strip()
        gemini_key = (
            values.get("GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()

        configured_model = _get_env("LLM_GATEWAY_MODEL", "").strip()
        if configured_model:
            model = configured_model
        elif openrouter_key:
            model = "portfolio-free"
        elif gemini_key:
            model = "portfolio-gemini"
        else:
            raise RuntimeError(
                f"No LiteLLM upstream provider credential is configured in "
                f"{_POC_DIR / '.env'}. Add OPENROUTER_API_KEY or GEMINI_API_KEY "
                "to use the litellm backend. LITELLM_MASTER_KEY / "
                "LITELLM_API_KEY only authenticate the app to the local "
                "LiteLLM proxy; they are not an OpenRouter/Gemini provider key."
            )

        if model == "portfolio-free" and not openrouter_key:
            raise RuntimeError(
                "LLM_GATEWAY_MODEL=portfolio-free requires OPENROUTER_API_KEY "
                f"in {_POC_DIR / '.env'}. Either add that key or remove "
                "LLM_GATEWAY_MODEL so the app can select an available provider."
            )
        if model == "portfolio-gemini" and not gemini_key:
            raise RuntimeError(
                "LLM_GATEWAY_MODEL=portfolio-gemini requires GEMINI_API_KEY "
                f"in {_POC_DIR / '.env'}. Either add that key or remove "
                "LLM_GATEWAY_MODEL so the app can select an available provider."
            )

        return ChatOpenAI(
            model=model,
            base_url=_get_env("LLM_GATEWAY_URL", "http://litellm:4000/v1"),
            api_key=_get_env("LLM_GATEWAY_API_KEY", ""),
            temperature=0,
        )

    if backend == "vllm":
        # vLLM exposes an OpenAI-compatible API. The endpoint, model and
        # API key remain environment-configurable so the same application can
        # run against a local container or any compatible remote endpoint.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_get_env("VLLM_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=_get_env("VLLM_BASE_URL", "http://vllm:8000/v1"),
            api_key=_get_env("VLLM_API_KEY", "local"),
            temperature=0,
        )

    raise ValueError(
        f"Unknown backend {backend!r}; expected 'litellm', 'huggingface', 'gemini-api', or 'vllm'"
    )


def content_to_text(content: Any) -> str:
    """Normalize a LangChain message's `.content` to plain text.

    `BaseMessage.content` is documented as `str | list[str | dict]`, not always
    a plain string — confirmed live, not theoretical: Gemini via
    `langchain_google_genai.ChatGoogleGenerativeAI` (the `"gemini-api"` default
    backend) returns a list of content blocks
    (`[{"type": "text", "text": "...", "extras": {...}}, ...]`) even for a
    plain-text answer with no multimodal content involved. Every place that
    treats `.content` as text (streaming accumulation, the final answer, the
    `[EVID: E<n>]` citation regex) raises `TypeError: expected string or
    bytes-like object, got 'list'` without this. HuggingFace/Bedrock's
    integrations return plain strings, so it isn't hit on those backends.

    Lives here rather than in one script because all four entry points read
    `result["messages"][-1].content` and every one of them needs it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def build_evidence_registry(chunks: list[RetrievedChunk]) -> dict:
    """Build the {eid_to_doc, eid_to_content} registry verifier.py expects,
    from the same RetrievedChunk list search_knowledge_base wrote to disk."""
    return {
        "eid_to_doc": {c.evidence_id: c.document_id for c in chunks},
        "eid_to_content": {c.evidence_id: c.content for c in chunks},
    }
