from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import aiosqlite
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.multi_agent_orchestrator import astream_langgraph_turn, build_langgraph_agent
from app.gateway_handoff import (
    GatewayHandoffError,
    exchange_handoff_code,
    revoke_gateway_session,
)
from app.api_store import EvidenceFlowStore, SessionStoreUnavailable
from retrieval.rag_pipeline import (
    DEFAULT_BACKEND,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    delete_document,
    ingest_document,
)
from shared.document_access import filter_documents_for_owner
from shared.rate_limit import (
    RateLimitBackendUnavailable,
    RateLimitExceeded,
    owner_rate_limiter,
)
from shared.security_logging import log_safe_failure
from shared.upload_security import validate_declared_upload_size

LOGGER = logging.getLogger("evidenceflow.api")
COOKIE_NAME = "evidenceflow_session"
CHAT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
EVIDENCE_TAG_PATTERN = re.compile(r"\s*\[EVID:\s*(E\d+)\]", re.IGNORECASE)
MAX_MESSAGES_PER_CHAT = 100
MAX_CHATS_PER_OWNER = 100
MAX_AGENT_CACHE = 64

# Public progress events are a deliberately small, static vocabulary. The API
# never relays arbitrary LangGraph event names, prompts, intermediate outputs
# or model reasoning to client-facing activity views.
PUBLIC_TASKS = {
    "route": ("Plan research", "Classify the request and choose relevant research branches."),
    "retrieve_kb": ("Search knowledge base", "Retrieve relevant passages from the client's indexed documents."),
    "analyze_chunk": ("Analyze evidence", "Inspect retrieved passages and extract relevant supporting facts."),
    "summarize_document": ("Summarize documents", "Consolidate evidence across related document sections."),
    "research_web": ("Search public sources", "Gather relevant external information when the request needs it."),
    "research_primary_source": ("Check primary sources", "Consult approved primary-source data connectors."),
    "synthesize": ("Synthesize answer", "Combine selected evidence into a grounded response."),
}
PUBLIC_TASK_ID_PATTERN = re.compile(r"^(?:route|retrieve_kb|analyze_chunk|summarize_document|research_web|research_primary_source|synthesize)-[1-9][0-9]{0,5}$")

store = EvidenceFlowStore()
checkpointer: AsyncSqliteSaver | None = None
_checkpointer_connection: aiosqlite.Connection | None = None

# Graphs are cached per opaque app session + thread. Each graph gets a
# thread-specific mutable scope snapshot so document ownership remains
# isolated while newly ingested files become visible on the next turn.
_AGENT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_AGENT_BUILD_LOCK = asyncio.Lock()
_THREAD_LOCKS: dict[str, asyncio.Lock] = {}


class SessionExchangeRequest(BaseModel):
    code: str = Field(min_length=32, max_length=128)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)


def _is_production() -> bool:
    return (
        os.getenv("APP_ENV", "").strip().lower() in {"prod", "production"}
        or os.getenv("RENDER", "").strip().lower() in {"1", "true"}
        or bool(os.getenv("K_SERVICE"))
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_for_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(payload), separators=(',', ':'))}\n\n"


def _public_task_update(event: Any) -> dict[str, Any] | None:
    """Return a minimal, client-safe task lifecycle event."""
    if not isinstance(event, dict):
        return None
    name = event.get("name")
    details = PUBLIC_TASKS.get(name) if isinstance(name, str) else None
    task_id = event.get("id")
    status = event.get("status")
    if (
        details is None
        or not isinstance(task_id, str)
        or not PUBLIC_TASK_ID_PATTERN.fullmatch(task_id)
        or status not in {"running", "completed", "failed"}
    ):
        return None
    label, description = details
    return {
        "id": task_id,
        "name": name,
        "label": label,
        "description": description,
        "status": status,
    }


def _strip_evidence_tags(value: str) -> str:
    """Convert trusted evidence markers into navigable links, not plain text."""
    return EVIDENCE_TAG_PATTERN.sub(
        lambda match: f" [{match.group(1).upper()}](#evidence-{match.group(1).upper()})",
        str(value or ""),
    ).strip()


def _graph_expanded_chunk_count(chunks: list[Any] | None) -> int:
    """Count graph-expanded chunks from either internal dataclasses or API dicts."""
    count = 0
    for chunk in chunks or []:
        metadata = (
            chunk.get("meta_data")
            if isinstance(chunk, dict)
            else getattr(chunk, "meta_data", None)
        )
        if isinstance(metadata, dict) and metadata.get("graph_expanded") is True:
            count += 1
    return count


def _safe_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """Return only bounded, UI-relevant answer metadata; never persist owner IDs."""
    safe = {
        key: outcome[key]
        for key in (
            "abstained",
            "abstention_reason",
            "grounding_status",
            "quality",
            "trust",
            "used_knowledge_base",
            "used_web",
            "used_primary_source",
            "token_usage",
            "verified_count",
            "proposed_count",
        )
        if key in outcome
    }
    safe["final_answer"] = _strip_evidence_tags(outcome.get("final_answer", ""))

    chunks = []
    for chunk in outcome.get("chunks") or []:
        item = asdict(chunk) if is_dataclass(chunk) else dict(chunk)
        metadata = item.get("meta_data") if isinstance(item.get("meta_data"), dict) else {}
        item["meta_data"] = {
            key: metadata[key]
            for key in (
                "source_name", "file_name", "document_name", "page_no",
                "section", "modality", "title", "url",
                "graph_expanded", "graph_hop", "graph_shared_entities",
            )
            if key in metadata
        }
        item.pop("owner_scope", None)
        item["content"] = str(item.get("content") or "")[:1400]
        chunks.append(item)
    safe["chunks"] = chunks[:8]

    web_sources = []
    for source in outcome.get("web_sources") or []:
        source_data = asdict(source) if is_dataclass(source) else (
            dict(source) if isinstance(source, dict) else jsonable_encoder(source)
        )
        if not isinstance(source_data, dict):
            continue
        web_sources.append({
            key: str(source_data.get(key) or "")[:1200]
            for key in ("title", "url", "snippet", "domain", "source_name")
            if source_data.get(key) is not None
        })
    safe["web_sources"] = web_sources[:8]

    delegations = []
    for step in outcome.get("delegations") or outcome.get("agent_trace") or []:
        step_data = asdict(step) if is_dataclass(step) else (
            dict(step) if isinstance(step, dict) else jsonable_encoder(step)
        )
        if not isinstance(step_data, dict):
            continue
        delegations.append({
            key: str(step_data.get(key) or "")[:500]
            for key in ("agent", "subagent_type", "description", "responsibility", "result")
            if step_data.get(key) is not None
        })
    safe["delegations"] = delegations[:12]
    return jsonable_encoder(safe)


def _load_workspace(owner_scope: str) -> dict[str, Any]:
    try:
        return store.get_workspace(owner_scope)
    except SessionStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


def _save_workspace(owner_scope: str, state: dict[str, Any]) -> None:
    try:
        store.save_workspace(owner_scope, state)
    except SessionStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


def _find_chat(state: dict[str, Any], thread_id: str) -> dict[str, Any] | None:
    if not CHAT_ID_PATTERN.fullmatch(str(thread_id or "")):
        return None
    return next(
        (chat for chat in state["chats"] if chat.get("thread_id") == thread_id),
        None,
    )


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Expose only the document metadata the browser needs to display or remove it."""
    public_fields = (
        "name",
        "document_id",
        "chunk_count",
        "scope",
        "created_at",
        "visual_pages_processed",
        "visual_pages_skipped",
        "modalities",
    )
    return {key: document[key] for key in public_fields if key in document}


def _public_chat(chat: dict[str, Any]) -> dict[str, Any]:
    """Project server-side chat state into the browser-safe response contract."""
    public = {
        key: chat[key]
        for key in ("thread_id", "title", "messages", "created_at", "updated_at")
        if key in chat
    }
    public["uploaded_docs"] = [
        _public_document(document)
        for document in chat.get("uploaded_docs", [])
    ]
    return public


def _session_from_request(request: Request) -> tuple[str, dict[str, Any]]:
    session_id = request.cookies.get(COOKIE_NAME, "")
    try:
        session = store.get_session(session_id)
    except SessionStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    if session is None:
        raise HTTPException(401, "Your EvidenceFlow session is missing or expired. Launch it again from the portfolio.")
    return session_id, session


def require_session(request: Request) -> tuple[str, dict[str, Any]]:
    return _session_from_request(request)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global checkpointer, _checkpointer_connection
    history_dir = Path(os.getenv("EVIDENCEFLOW_DATA_DIR", "chat_history"))
    history_dir.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(str(history_dir / "checkpoints.db"))
    saver = AsyncSqliteSaver(connection)
    await saver.setup()
    _checkpointer_connection = connection
    checkpointer = saver
    yield
    _AGENT_CACHE.clear()
    _THREAD_LOCKS.clear()
    checkpointer = None
    _checkpointer_connection = None
    await connection.close()


app = FastAPI(
    title="EvidenceFlow API",
    version="1.0.0",
    description="API-only, client-scoped research and document intelligence service. Supports secure document ingestion, sparse-first retrieval, evidence verification and streamed research responses.",
    lifespan=lifespan,
)

_DEFAULT_DEV_ORIGINS = (
    ""
    if _is_production()
    else "http://localhost:3000,http://127.0.0.1:3000"
)
_ALLOWED_ORIGINS = [
    value.strip().rstrip("/")
    for value in os.getenv("CORS_ORIGINS", _DEFAULT_DEV_ORIGINS).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Requested-With"],
)


@app.middleware("http")
async def protect_cookie_mutations(request: Request, call_next):
    if request.method in {"POST", "DELETE", "PUT", "PATCH"} and request.url.path.startswith("/api/v1/"):
        origin = request.headers.get("origin")
        if origin:
            request_origin = origin.rstrip("/")
            request_host = request.headers.get("host", "").lower()
            origin_host = (urlsplit(request_origin).netloc or "").lower()
            same_origin = bool(request_host and origin_host == request_host)
            if not same_origin and request_origin not in _ALLOWED_ORIGINS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Request origin is not permitted."},
                )
    result = await call_next(request)
    if request.url.path.startswith("/api/v1/"):
        result.headers["Cache-Control"] = "no-store"
        result.headers["Pragma"] = "no-cache"
    result.headers.setdefault("X-Content-Type-Options", "nosniff")
    result.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    result.headers.setdefault("X-Frame-Options", "DENY")
    return result


@app.get("/health")
def health():
    return {"status": "ok", "service": "evidenceflow-api", "version": app.version}


@app.get("/ready")
def ready():
    if checkpointer is None:
        raise HTTPException(503, "Conversation storage is not ready.")
    try:
        store.check_ready()
    except SessionStoreUnavailable:
        raise HTTPException(503, "Workspace storage is not ready.") from None
    return {"status": "ready", "workspace_store": "ok", "checkpointer": "ok"}


@app.get("/_stcore/health")
def legacy_render_health():
    """Compatibility alias for the existing Render health-check setting.

    This endpoint only reports API readiness. It does not mount or expose
    the retired Streamlit interface.
    """
    return ready()


@app.post("/api/v1/session/exchange")
async def exchange_session(payload: SessionExchangeRequest, request: Request):
    identity_header_names = (
        "X-Studio-Workspace-Client-Id",
        "X-Studio-Workspace-Timestamp",
        "X-Studio-Workspace-Signature",
    )
    identity_headers = {
        name: request.headers.get(name, "").strip()
        for name in identity_header_names
    }
    trusted_identity = identity_headers if any(identity_headers.values()) else None
    try:
        exchanged = await asyncio.to_thread(
            exchange_handoff_code,
            payload.code,
            project="evidenceflow",
            identity_headers=trusted_identity,
        )
        session_id = store.create_session(
            access_token=exchanged["access_token"],
            owner_scope=exchanged["owner_scope"],
            provider=exchanged.get("provider", ""),
            model=exchanged.get("model", ""),
            expires_in=exchanged["expires_in"],
        )
    except GatewayHandoffError as exc:
        raise HTTPException(401, str(exc)) from None
    except SessionStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from None

    result = JSONResponse(
        {
            "active": True,
            "project": "evidenceflow",
            "provider": exchanged.get("provider", ""),
            "model": exchanged.get("model", ""),
            "expires_at": int(time.time()) + exchanged["expires_in"],
        }
    )
    result.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        max_age=exchanged["expires_in"],
        httponly=True,
        secure=_is_production(),
        samesite="lax",
        path="/",
    )
    return result


@app.get("/api/v1/session")
def session_status(session=Depends(require_session)):
    _, details = session
    return {
        "active": True,
        "project": "evidenceflow",
        "provider": details.get("provider", ""),
        "model": details.get("model", ""),
        "expires_at": details["expires_at"],
    }


@app.post("/api/v1/session/revoke")
def revoke_session(request: Request):
    session_id = request.cookies.get(COOKIE_NAME, "")
    if session_id:
        try:
            session = store.get_session(session_id)
            if session is not None:
                # The Gateway child session is the actual provider credential.
                # Keep the encrypted local record and browser cookie until the
                # Gateway confirms revocation so a transient failure is retryable.
                revoke_gateway_session(session["access_token"])
            store.revoke_session(session_id)
        except (GatewayHandoffError, SessionStoreUnavailable) as exc:
            raise HTTPException(503, str(exc)) from None
        session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        for key in list(_AGENT_CACHE):
            if key.startswith(session_hash + ":"):
                _AGENT_CACHE.pop(key, None)

    result = JSONResponse({"revoked": True}, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    result.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=_is_production(), samesite="lax")
    return result


@app.get("/api/v1/chats")
def list_chats(session=Depends(require_session)):
    _, details = session
    state = _load_workspace(details["owner_scope"])
    chats = [
        {
            "thread_id": chat["thread_id"],
            "title": chat.get("title") or "New conversation",
            "updated_at": chat.get("updated_at"),
            "message_count": len(chat.get("messages") or []),
        }
        for chat in state["chats"]
    ]
    chats.sort(key=lambda chat: chat.get("updated_at") or "", reverse=True)
    return {"data": chats}


@app.post("/api/v1/chats", status_code=201)
def create_chat(session=Depends(require_session)):
    _, details = session
    state = _load_workspace(details["owner_scope"])
    if len(state["chats"]) >= MAX_CHATS_PER_OWNER:
        raise HTTPException(409, "Conversation limit reached. Delete an older conversation to continue.")
    chat = {
        "thread_id": uuid.uuid4().hex,
        "title": "New conversation",
        "messages": [],
        "uploaded_docs": [],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    state["chats"].append(chat)
    state["chats"] = state["chats"][-MAX_CHATS_PER_OWNER:]
    _save_workspace(details["owner_scope"], state)
    return chat


@app.get("/api/v1/chats/{thread_id}")
def get_chat(thread_id: str, session=Depends(require_session)):
    _, details = session
    state = _load_workspace(details["owner_scope"])
    chat = _find_chat(state, thread_id)
    if chat is None:
        raise HTTPException(404, "Conversation not found.")
    return _public_chat(chat)


@app.delete("/api/v1/chats/{thread_id}")
async def delete_chat(thread_id: str, session=Depends(require_session)):
    session_id, details = session
    owner_scope = details["owner_scope"]
    state = _load_workspace(owner_scope)
    chat = _find_chat(state, thread_id)
    if chat is None:
        raise HTTPException(404, "Conversation not found.")

    attached = filter_documents_for_owner(chat.get("uploaded_docs", []), owner_scope)
    try:
        for document in attached:
            await delete_document(document["document_id"], owner_scope=owner_scope)
        if checkpointer is not None:
            await checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        incident_id = log_safe_failure(LOGGER, "delete_chat", exc)
        raise HTTPException(502, f"Conversation cleanup failed. Reference: {incident_id}") from None

    state["chats"] = [item for item in state["chats"] if item.get("thread_id") != thread_id]
    _save_workspace(owner_scope, state)
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    _AGENT_CACHE.pop(f"{session_hash}:{thread_id}", None)
    _THREAD_LOCKS.pop(f"{owner_scope}:{thread_id}", None)
    return {"deleted": True, "thread_id": thread_id}


@app.get("/api/v1/documents")
def list_documents(
    scope: Literal["global", "chat", "all"] = "all",
    thread_id: str | None = None,
    session=Depends(require_session),
):
    _, details = session
    state = _load_workspace(details["owner_scope"])
    documents = []
    if scope in {"global", "all"}:
        documents.extend(state["global_documents"])
    if scope in {"chat", "all"}:
        if thread_id:
            chat = _find_chat(state, thread_id)
            if chat is None:
                raise HTTPException(404, "Conversation not found.")
            documents.extend(filter_documents_for_owner(chat.get("uploaded_docs", []), details["owner_scope"]))
        elif scope == "chat":
            raise HTTPException(422, "thread_id is required for chat-scoped documents.")
    return {"data": [_public_document(document) for document in documents]}


@app.post("/api/v1/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    scope: Literal["global", "chat"] = Form("global"),
    thread_id: str | None = Form(None),
    session=Depends(require_session),
):
    _, details = session
    owner_scope = details["owner_scope"]
    state = _load_workspace(owner_scope)
    target_chat = None
    if scope == "chat":
        target_chat = _find_chat(state, thread_id or "")
        if target_chat is None:
            raise HTTPException(404, "Conversation not found.")

    source_name = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1]
    try:
        validate_declared_upload_size(getattr(file, "size", None))
        content = await file.read()
        if not content:
            raise ValueError("Uploaded document is empty.")
        if len(content) > 200 * 1024 * 1024:
            raise ValueError("Uploaded document exceeds the configured size limit.")
        target_documents = state["global_documents"] if scope == "global" else target_chat["uploaded_docs"]
        content_hash = hashlib.sha256(content).hexdigest()
        duplicate = next(
            (item for item in target_documents if item.get("content_hash") == content_hash),
            None,
        )
        if duplicate:
            return {"document": _public_document(duplicate), "duplicate": True}
        result = await ingest_document(
            content,
            source_name=source_name,
            owner_scope=owner_scope,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception as exc:
        incident_id = log_safe_failure(LOGGER, "document_ingest", exc)
        raise HTTPException(502, f"Document processing failed. Reference: {incident_id}") from None
    finally:
        await file.close()

    document = {
        "name": source_name,
        "document_id": result["document_id"],
        "chunk_count": result["chunk_count"],
        "content_hash": content_hash,
        "owner_scope": owner_scope,
        "scope": scope,
        "visual_pages_processed": result.get("visual_pages_processed", 0),
        "visual_pages_skipped": result.get("visual_pages_skipped", 0),
        "modalities": result.get("modalities", []),
        "created_at": _utc_now(),
    }
    target_documents.append(document)
    _save_workspace(owner_scope, state)
    return {"document": _public_document(document), "duplicate": False}


@app.delete("/api/v1/documents/{document_id}")
async def remove_document(
    document_id: str,
    scope: Literal["global", "chat"] = "global",
    thread_id: str | None = None,
    session=Depends(require_session),
):
    _, details = session
    owner_scope = details["owner_scope"]
    state = _load_workspace(owner_scope)
    if scope == "global":
        target_documents = state["global_documents"]
    else:
        chat = _find_chat(state, thread_id or "")
        if chat is None:
            raise HTTPException(404, "Conversation not found.")
        target_documents = chat["uploaded_docs"]

    document = next(
        (item for item in filter_documents_for_owner(target_documents, owner_scope)
         if item.get("document_id") == document_id),
        None,
    )
    if document is None:
        raise HTTPException(404, "Document not found.")
    try:
        await delete_document(document_id, owner_scope=owner_scope)
    except Exception as exc:
        incident_id = log_safe_failure(LOGGER, "document_delete", exc)
        raise HTTPException(502, f"Document removal failed. Reference: {incident_id}") from None

    target_documents[:] = [item for item in target_documents if item.get("document_id") != document_id]
    _save_workspace(owner_scope, state)
    return {"deleted": True, "document_id": document_id}


def _thread_lock(owner_scope: str, thread_id: str) -> asyncio.Lock:
    key = f"{owner_scope}:{thread_id}"
    if key not in _THREAD_LOCKS:
        _THREAD_LOCKS[key] = asyncio.Lock()
    return _THREAD_LOCKS[key]


async def _get_or_build_agent(
    *,
    session_id: str,
    details: dict[str, Any],
    thread_id: str,
    scope_snapshot: dict[str, Any],
):
    if checkpointer is None:
        raise HTTPException(503, "Conversation storage is not ready.")
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    cache_key = f"{session_hash}:{thread_id}"
    async with _AGENT_BUILD_LOCK:
        cached = _AGENT_CACHE.get(cache_key)
        if cached is not None:
            _AGENT_CACHE.move_to_end(cache_key)
            cached["scope_snapshot"].update(scope_snapshot)
            return cached["agent"]
        built = await asyncio.to_thread(
            build_langgraph_agent,
            document_ids_provider=lambda: list(scope_snapshot["ids"]),
            document_names_provider=lambda: dict(scope_snapshot["names"]),
            owner_scope_provider=lambda: details["owner_scope"],
            backend=DEFAULT_BACKEND,
            checkpointer=checkpointer,
            gateway_token=details["access_token"],
        )
        _AGENT_CACHE[cache_key] = {
            "agent": built,
            "scope_snapshot": scope_snapshot,
        }
        while len(_AGENT_CACHE) > MAX_AGENT_CACHE:
            _AGENT_CACHE.popitem(last=False)
        return built


@app.post("/api/v1/chats/{thread_id}/messages")
async def stream_chat_message(
    thread_id: str,
    payload: MessageRequest,
    session=Depends(require_session),
):
    session_id, details = session
    owner_scope = details["owner_scope"]
    state = _load_workspace(owner_scope)
    chat = _find_chat(state, thread_id)
    if chat is None:
        raise HTTPException(404, "Conversation not found.")
    message = payload.message.strip()
    if not message:
        raise HTTPException(422, "Message cannot be empty.")
    try:
        await asyncio.to_thread(
            owner_rate_limiter.enforce,
            owner_scope,
            "chat",
            limit=30,
            window_seconds=60,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            429,
            "EvidenceFlow message limit reached. Please retry shortly.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from None
    except RateLimitBackendUnavailable:
        raise HTTPException(503, "Request quota could not be verified. Please retry shortly.") from None

    async def event_stream():
        lock = _thread_lock(owner_scope, thread_id)
        async with lock:
            live_state = _load_workspace(owner_scope)
            live_chat = _find_chat(live_state, thread_id)
            if live_chat is None:
                yield _json_for_sse("error", {"message": "Conversation not found."})
                return

            user_message = {
                "id": uuid.uuid4().hex,
                "role": "user",
                "content": message,
                "created_at": _utc_now(),
            }
            live_chat.setdefault("messages", []).append(user_message)
            live_chat["messages"] = live_chat["messages"][-MAX_MESSAGES_PER_CHAT:]
            if not live_chat.get("title") or live_chat["title"] == "New conversation":
                live_chat["title"] = message[:50] + ("…" if len(message) > 50 else "")
            live_chat["updated_at"] = _utc_now()
            _save_workspace(owner_scope, live_state)

            global_docs = filter_documents_for_owner(live_state["global_documents"], owner_scope)
            chat_docs = filter_documents_for_owner(live_chat.get("uploaded_docs", []), owner_scope)
            documents = global_docs + chat_docs
            scope_snapshot = {
                "ids": [item["document_id"] for item in documents],
                "names": {item["document_id"]: item["name"] for item in documents},
            }
            started = time.perf_counter()
            try:
                agent = await _get_or_build_agent(
                    session_id=session_id,
                    details=details,
                    thread_id=thread_id,
                    scope_snapshot=scope_snapshot,
                )
                final_outcome = None
                async with asyncio.timeout(DEFAULT_TURN_TIMEOUT_SECONDS):
                    async for event in astream_langgraph_turn(agent, message, thread_id=thread_id):
                        if event.get("type") == "status":
                            yield _json_for_sse("status", {"text": event.get("text", "")})
                        elif event.get("type") == "task":
                            task = _public_task_update(event.get("task"))
                            if task is not None:
                                yield _json_for_sse("task", {"task": task})
                        elif event.get("type") == "token":
                            yield _json_for_sse("token", {"text": event.get("text", "")})
                        elif event.get("type") == "final":
                            final_outcome = event

                if final_outcome is None:
                    raise RuntimeError("Research stream ended without a final result.")

                outcome = _safe_outcome(final_outcome)
                assistant_text = outcome.get("final_answer", "")
                assistant_message = {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": assistant_text,
                    "meta": outcome,
                    "created_at": _utc_now(),
                }
                live_chat["messages"].append(assistant_message)
                live_chat["messages"] = live_chat["messages"][-MAX_MESSAGES_PER_CHAT:]
                live_chat["updated_at"] = _utc_now()
                _save_workspace(owner_scope, live_state)

                try:
                    from shared.evaluation_harness import (
                        build_runtime_evaluation_metrics,
                        record_event,
                    )

                    duration_ms = (time.perf_counter() - started) * 1000
                    model_label = details.get("model") or None
                    record_event(
                        event_type="run.completed",
                        status="warning" if outcome.get("abstained") else "success",
                        workflow="rag_query",
                        duration_ms=duration_ms,
                        model=model_label,
                        metrics={
                            "retrieved_chunks": len(outcome.get("chunks") or []),
                            "graph_expanded_chunks": _graph_expanded_chunk_count(
                                outcome.get("chunks")
                            ),
                            "web_sources": len(outcome.get("web_sources") or []),
                            "used_knowledge_base": bool(outcome.get("used_knowledge_base")),
                            "used_primary_source": bool(outcome.get("used_primary_source")),
                        },
                    )

                    runtime_metrics = build_runtime_evaluation_metrics(final_outcome)
                    if runtime_metrics:
                        record_event(
                            event_type="evaluation.completed",
                            status="success",
                            workflow="rag_query",
                            duration_ms=duration_ms,
                            model=model_label,
                            model_version=model_label,
                            experiment_id=os.getenv("EVALUATION_EXPERIMENT_ID") or None,
                            dataset_version=os.getenv("EVALUATION_DATASET_VERSION") or None,
                            prompt_version=os.getenv("EVALUATION_PROMPT_VERSION") or None,
                            config_version=os.getenv("EVALUATION_CONFIG_VERSION") or None,
                            metrics=runtime_metrics,
                        )

                    # Optional external judging runs after the user-facing answer
                    # has been persisted. It uses the short-lived Gateway session
                    # and never sends source text, prompts or answers to the harness.
                    from retrieval.rag_pipeline import _get_env
                    from shared.ragas_evaluation import schedule_ragas_evaluation

                    retrieved_contexts = [
                        str(chunk.get("content") or "")[:1400]
                        for chunk in (outcome.get("chunks") or [])[:8]
                        if isinstance(chunk, dict) and chunk.get("content")
                    ]
                    schedule_ragas_evaluation(
                        user_input=message,
                        response=str(outcome.get("final_answer") or ""),
                        retrieved_contexts=retrieved_contexts,
                        gateway_token=str(details.get("access_token") or ""),
                        gateway_url=(
                            _get_env("LLM_GATEWAY_URL", "").strip()
                            or _get_env("LLM_BASE_URL", "").strip()
                        ),
                        model_label=model_label,
                        experiment_id=os.getenv("EVALUATION_EXPERIMENT_ID") or None,
                        dataset_version=os.getenv("EVALUATION_DATASET_VERSION") or None,
                        prompt_version=os.getenv("EVALUATION_PROMPT_VERSION") or None,
                        config_version=os.getenv("EVALUATION_CONFIG_VERSION") or None,
                    )
                except Exception as exc:
                    LOGGER.debug(
                        "optional_evaluation_reporting_skipped error_type=%s",
                        type(exc).__name__,
                    )
                yield _json_for_sse("final", {"message": assistant_message, "outcome": outcome})
            except asyncio.TimeoutError:
                message_text = f"Research exceeded the {DEFAULT_TURN_TIMEOUT_SECONDS}-second time limit. Please retry."
                assistant_message = {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": message_text,
                    "kind": "turn_error",
                    "created_at": _utc_now(),
                }
                live_chat["messages"].append(assistant_message)
                live_chat["messages"] = live_chat["messages"][-MAX_MESSAGES_PER_CHAT:]
                live_chat["updated_at"] = _utc_now()
                _save_workspace(owner_scope, live_state)
                yield _json_for_sse("error", {"message": message_text})
            except Exception as exc:
                incident_id = log_safe_failure(LOGGER, "langgraph_turn", exc)
                message_text = f"Research failed. Please retry. Reference: {incident_id}"
                assistant_message = {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": message_text,
                    "kind": "turn_error",
                    "created_at": _utc_now(),
                }
                live_chat["messages"].append(assistant_message)
                live_chat["messages"] = live_chat["messages"][-MAX_MESSAGES_PER_CHAT:]
                live_chat["updated_at"] = _utc_now()
                _save_workspace(owner_scope, live_state)
                yield _json_for_sse("error", {"message": message_text})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", include_in_schema=False)
def api_root():
    """Return service metadata; EvidenceFlow is an API-only deployment."""
    return {
        "service": "evidenceflow-api",
        "status": "ok",
        "version": app.version,
        "api_base_path": "/api/v1",
        "openapi_url": "/openapi.json",
        "docs_url": "/docs",
        "health_url": "/health",
        "readiness_url": "/ready",
        "capabilities": [
            "owner-scoped document ingestion",
            "sparse-first retrieval and reranking",
            "evidence-backed research with citation verification",
            "server-sent event responses",
            "client-safe live task lifecycle events",
        ],
        "document_types": [
            "pdf", "doc", "docx", "xls", "xlsx", "xlsm", "xlsb",
            "ods", "ppt", "pptx", "odp", "odt", "rtf", "txt", "md", "csv",
            "tsv", "json", "xml", "yaml", "toml", "ini", "log",
            "html", "htm", "eml", "sql", "py", "js", "ts", "css",
            "sh", "conf", "png", "jpg", "jpeg", "webp", "bmp",
            "tif", "tiff",
        ],
    }
