from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CHAT_DIR = ROOT / "chat_history"
CHAT_DIR.mkdir(exist_ok=True)

from agents.custom_langgraph_poc import astream_langgraph_turn, build_langgraph_agent  # noqa: E402
from retrieval.rag_pipeline import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    delete_document,
    ingest_document,
)

app = FastAPI(title="EvidenceFlow API", version="1.0.0")

# One shared SQLite checkpointer across the Render process. Thread IDs are unique,
# so multiple browser sessions can safely share the same checkpoint store.
_checkpointer = None
_agent_cache: dict[str, dict] = {}
_agent_lock = threading.Lock()
_session_state: dict[str, dict] = {}


class ChatRequest(BaseModel):
    thread_id: str
    query: str


def _token_from_header(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A Portfolio BYOK session is required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A Portfolio BYOK session is required.")
    return token


def _session_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]


def _state(token: str) -> dict:
    key = _session_key(token)
    return _session_state.setdefault(key, {"global_docs": [], "chat_docs": {}})


def _scope_ids(token: str, thread_id: str) -> list[str]:
    state = _state(token)
    ids = [d["document_id"] for d in state["global_docs"]]
    ids.extend(d["document_id"] for d in state["chat_docs"].get(thread_id, []))
    # stable unique ordering
    return list(dict.fromkeys(ids))


def _scope_names(token: str, thread_id: str) -> dict[str, str]:
    state = _state(token)
    docs = list(state["global_docs"]) + list(state["chat_docs"].get(thread_id, []))
    return {d["document_id"]: d["name"] for d in docs}


async def _get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn_path = str(CHAT_DIR / "checkpoints.db")
        import aiosqlite
        conn = await aiosqlite.connect(conn_path)
        _checkpointer = AsyncSqliteSaver(conn)
        await _checkpointer.setup()
    return _checkpointer


def _get_agent(token: str):
    key = _session_key(token)
    with _agent_lock:
        built = _agent_cache.get(key)
    if built is not None:
        return built
    # Build outside the lock because checkpointer setup is async.
    return None


async def _ensure_agent(token: str) -> dict:
    key = _session_key(token)
    existing = _agent_cache.get(key)
    if existing:
        return existing
    checkpointer = await _get_checkpointer()
    loop_lock = getattr(_ensure_agent, "_lock", None)
    if loop_lock is None:
        loop_lock = asyncio.Lock()
        setattr(_ensure_agent, "_lock", loop_lock)
    async with loop_lock:
        existing = _agent_cache.get(key)
        if existing:
            return existing
        built = build_langgraph_agent(
            document_ids_provider=lambda: _current_ids(token),
            document_names_provider=lambda: _current_names(token),
            backend=DEFAULT_BACKEND,
            checkpointer=checkpointer,
            gateway_token=token,
        )
        _agent_cache[key] = built
        return built


# These provider closures always resolve the latest thread. The per-request
# task sets the active thread in session-local context below.
import contextvars
_active_thread: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_thread", default=""
)


def _current_ids(token: str) -> list[str]:
    return _scope_ids(token, _active_thread.get())


def _current_names(token: str) -> dict[str, str]:
    return _scope_names(token, _active_thread.get())


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "evidenceflow"}


@app.get("/api/session")
async def session_info(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    return {"authenticated": True, "backend": DEFAULT_BACKEND, "has_session": bool(token)}


@app.get("/api/documents")
async def list_documents(authorization: str | None = Header(default=None), thread_id: str = ""):
    token = _token_from_header(authorization)
    state = _state(token)
    return {
        "global": state["global_docs"],
        "chat": state["chat_docs"].get(thread_id, []),
        "scope_ids": _scope_ids(token, thread_id),
    }


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = "",
    scope: str = "chat",
    authorization: str | None = Header(default=None),
):
    token = _token_from_header(authorization)
    if scope not in {"global", "chat"}:
        raise HTTPException(status_code=400, detail="scope must be global or chat")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    result = await asyncio.wait_for(
        ingest_document(data, source_name=file.filename or "uploaded-document"),
        timeout=DEFAULT_TURN_TIMEOUT_SECONDS,
    )
    document_id = result["document_id"]
    entry = {
        "document_id": document_id,
        "name": file.filename or "uploaded-document",
        "content_hash": result.get("content_hash", ""),
    }
    state = _state(token)
    if scope == "global":
        state["global_docs"] = [d for d in state["global_docs"] if d["document_id"] != document_id]
        state["global_docs"].append(entry)
    else:
        state["chat_docs"].setdefault(thread_id, [])
        state["chat_docs"][thread_id] = [
            d for d in state["chat_docs"][thread_id] if d["document_id"] != document_id
        ]
        state["chat_docs"][thread_id].append(entry)
    return {"document": entry, "result": result}


@app.delete("/api/documents/{document_id}")
async def remove_document(
    document_id: str,
    thread_id: str = "",
    authorization: str | None = Header(default=None),
):
    token = _token_from_header(authorization)
    state = _state(token)
    try:
        await delete_document(document_id)
    finally:
        state["global_docs"] = [d for d in state["global_docs"] if d["document_id"] != document_id]
        state["chat_docs"][thread_id] = [
            d for d in state["chat_docs"].get(thread_id, []) if d["document_id"] != document_id
        ]
    return {"deleted": True, "document_id": document_id}


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    token = _token_from_header(authorization)
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if len(request.query) > 12000:
        raise HTTPException(status_code=413, detail="query is too long")

    built = await _ensure_agent(token)
    ctx = _active_thread.set(request.thread_id)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in astream_langgraph_turn(
                built, request.query.strip(), thread_id=request.thread_id
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload)}\n\n"
        finally:
            _active_thread.reset(ctx)
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve the Vite build from the same Render web service.
WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/{path:path}")
async def spa(path: str):
    if path.startswith("api/") or path == "healthz":
        return JSONResponse({"detail": "Not found"}, status_code=404)
    index = WEB_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"detail": "Frontend not built"}, status_code=503)
