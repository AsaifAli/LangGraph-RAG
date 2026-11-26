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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.multi_agent_orchestrator import astream_langgraph_turn, build_langgraph_agent
from app.gateway_handoff import GatewayHandoffError, exchange_handoff_code
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
EVIDENCE_TAG_PATTERN = re.compile(r"\s*\[EVID:\s*E\d+\]")
MAX_MESSAGES_PER_CHAT = 100
MAX_CHATS_PER_OWNER = 100
MAX_AGENT_CACHE = 64

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


def _strip_evidence_tags(value: str) -> str:
    return EVIDENCE_TAG_PATTERN.sub("", str(value or "")).strip()


def _safe_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """Convert graph results to bounded JSON metadata for the browser/history."""
    safe = dict(outcome)
    chunks = []
    for chunk in outcome.get("chunks") or []:
        item = asdict(chunk) if is_dataclass(chunk) else dict(chunk)
        item["content"] = str(item.get("content") or "")[:2400]
        chunks.append(item)
    safe["chunks"] = chunks[:12]
    safe["web_sources"] = (outcome.get("web_sources") or [])[:12]
    safe["final_answer"] = _strip_evidence_tags(outcome.get("final_answer", ""))
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
    description="Owner-scoped research workspace over the existing LangGraph and sparse-first RAG engine.",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = [
    value.strip().rstrip("/")
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
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
    return await call_next(request)


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


@app.post("/api/v1/session/exchange")
async def exchange_session(payload: SessionExchangeRequest):
    try:
        exchanged = await asyncio.to_thread(
            exchange_handoff_code,
            payload.code,
            project="evidenceflow",
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
            store.revoke_session(session_id)
        except SessionStoreUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        for key in list(_AGENT_CACHE):
            if key.startswith(session_hash + ":"):
                _AGENT_CACHE.pop(key, None)
    result = JSONResponse({"revoked": True})
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
    return chat


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
    return {"data": documents}


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
            return {"document": duplicate, "duplicate": True}
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
    return {"document": document, "duplicate": False}


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
                    from shared.evaluation_harness import record_event

                    record_event(
                        event_type="run.completed",
                        status="warning" if outcome.get("abstained") else "success",
                        workflow="rag_query",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        model=details.get("model") or None,
                        metrics={
                            "retrieved_chunks": len(outcome.get("chunks") or []),
                            "web_sources": len(outcome.get("web_sources") or []),
                            "used_knowledge_base": bool(outcome.get("used_knowledge_base")),
                            "used_primary_source": bool(outcome.get("used_primary_source")),
                        },
                    )
                except Exception:
                    LOGGER.debug("optional_evaluation_event_not_recorded")
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


# In the hosted layout the Vite build is served by the same API origin. This
# keeps the HttpOnly session cookie first-party and avoids exposing Gateway
# bearer tokens to browser storage or client-side JavaScript.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if (_FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.get("/", include_in_schema=False)
def frontend_root():
    index = _FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        status_code=503,
        content={"detail": "EvidenceFlow frontend has not been built."},
    )


@app.get("/{asset_path:path}", include_in_schema=False)
def frontend_fallback(asset_path: str):
    if asset_path.startswith("api/") or asset_path in {"health", "ready", "docs", "openapi.json"}:
        raise HTTPException(404, "Not found.")
    candidate = (_FRONTEND_DIST / asset_path).resolve()
    if candidate.is_file() and _FRONTEND_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = _FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(404, "Not found.")
