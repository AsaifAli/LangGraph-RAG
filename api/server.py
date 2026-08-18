from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CHAT_DIR = ROOT / "chat_history"
CHAT_DIR.mkdir(parents=True, exist_ok=True)

from agents.custom_langgraph_poc import (  # noqa: E402
    astream_langgraph_turn,
    build_langgraph_agent,
)
from config.loader import get_config  # noqa: E402
from retrieval.rag_pipeline import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    delete_document,
    ingest_document,
)

app = FastAPI(title="EvidenceFlow API", version="2.0.0")

_SESSION_RE = re.compile(r"\[EVID:\s*(E\d+)\]")
_ACTIVE_THREAD: contextvars.ContextVar[str] = contextvars.ContextVar("active_thread", default="")
_agent_cache: dict[str, dict[str, Any]] = {}
_agent_locks: dict[str, asyncio.Lock] = {}
_checkpointer = None
_checkpointer_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    thread_id: str
    query: str = Field(min_length=1, max_length=12000)


def _token_from_header(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A Portfolio BYOK session is required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A Portfolio BYOK session is required.")
    return token


def _session_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]


def _session_dir(token: str) -> Path:
    d = CHAT_DIR / "sessions" / _session_key(token)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _global_docs_path(token: str) -> Path:
    return _session_dir(token) / "uploaded_documents.json"


def _chat_path(token: str, thread_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", thread_id)
    return _session_dir(token) / f"{safe}.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_global_docs(token: str) -> list[dict[str, Any]]:
    return _read_json(_global_docs_path(token), [])


def _save_global_docs(token: str, docs: list[dict[str, Any]]) -> None:
    _write_json(_global_docs_path(token), docs)


def _load_chat(token: str, thread_id: str) -> dict[str, Any]:
    data = _read_json(_chat_path(token, thread_id), {})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("thread_id", thread_id)
    data.setdefault("title", None)
    data.setdefault("messages", [])
    data.setdefault("uploaded_docs", [])
    data.setdefault("updated_at", None)
    return data


def _save_chat(token: str, chat: dict[str, Any]) -> None:
    chat["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(_chat_path(token, chat["thread_id"]), chat)


def _list_chats(token: str) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []
    for path in _session_dir(token).glob("*.json"):
        if path.name == "uploaded_documents.json":
            continue
        data = _read_json(path, {})
        if isinstance(data, dict) and data.get("thread_id"):
            chats.append(
                {
                    "thread_id": data["thread_id"],
                    "title": data.get("title") or "Untitled chat",
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages") or []),
                }
            )
    chats.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return chats


def _scope_docs(token: str, thread_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_docs = _load_global_docs(token)
    chat = _load_chat(token, thread_id)
    return global_docs, list(chat.get("uploaded_docs") or [])


def _scope_ids(token: str, thread_id: str) -> list[str]:
    global_docs, chat_docs = _scope_docs(token, thread_id)
    return list(dict.fromkeys([d["document_id"] for d in [*global_docs, *chat_docs] if d.get("document_id")]))


def _scope_names(token: str, thread_id: str) -> dict[str, str]:
    global_docs, chat_docs = _scope_docs(token, thread_id)
    return {d["document_id"]: d["name"] for d in [*global_docs, *chat_docs]}


def _current_ids(token: str) -> list[str]:
    return _scope_ids(token, _ACTIVE_THREAD.get())


def _current_names(token: str) -> dict[str, str]:
    return _scope_names(token, _ACTIVE_THREAD.get())


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        raw = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
    except Exception:  # noqa: BLE001
        return {}


def _session_status(token: str) -> dict[str, Any]:
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    now = int(time.time())
    if isinstance(exp, (int, float)) and now >= int(exp):
        return {
            "status": "expired",
            "authenticated": False,
            "expires_at": int(exp),
            "provider": payload.get("provider"),
            "model": payload.get("model"),
        }
    return {
        "status": "active" if exp else "unknown",
        "authenticated": bool(payload),
        "expires_at": int(exp) if isinstance(exp, (int, float)) else None,
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "project": payload.get("project"),
    }


def _require_active_session(token: str) -> None:
    status = _session_status(token)
    if status.get("status") == "expired":
        raise HTTPException(status_code=401, detail="BYOK session expired. Return to the portfolio to start a new session.")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _display_answer(raw: str) -> str:
    return _SESSION_RE.sub("", raw or "").strip()


def _meta_from_final(final: dict[str, Any]) -> dict[str, Any]:
    raw = final.get("final_answer", "") or ""
    cited_ids = sorted(set(_SESSION_RE.findall(raw)))
    chunks = _jsonable(final.get("chunks") or [])
    if not final.get("used_knowledge_base"):
        chunks = []
        cited_ids = []
    return {
        "token_usage": _jsonable(final.get("token_usage")),
        "used_knowledge_base": bool(final.get("used_knowledge_base") and chunks),
        "chunks": chunks,
        "cited_evidence_ids": cited_ids,
        "grounding_status": final.get("grounding_status"),
        "reasons": _jsonable(final.get("reasons") or {}),
        "quality": _jsonable(final.get("quality") or {}),
        "abstained": bool(final.get("abstained")),
        "abstention_reason": final.get("abstention_reason"),
        "used_web": bool(final.get("used_web")),
        "web_sources": _jsonable(final.get("web_sources") or []),
        "used_primary_source": bool(final.get("used_primary_source")),
        "primary_source": _jsonable(final.get("primary_source") or {}),
        "plan": _jsonable(final.get("plan") or []),
        "retrieved_count": len(chunks),
        "verified_count": int(final.get("verified_count") or 0),
        "proposed_count": int(final.get("proposed_count") or 0),
    }


def _ensure_title(chat: dict[str, Any], query: str) -> None:
    if not chat.get("title"):
        chat["title"] = query.strip()[:50] + ("…" if len(query.strip()) > 50 else "")


async def _get_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    async with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

        conn = await aiosqlite.connect(str(CHAT_DIR / "checkpoints.db"))
        _checkpointer = AsyncSqliteSaver(conn)
        await _checkpointer.setup()
        return _checkpointer


async def _ensure_agent(token: str) -> dict[str, Any]:
    key = _session_key(token)
    if key in _agent_cache:
        return _agent_cache[key]
    lock = _agent_locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _agent_cache:
            return _agent_cache[key]
        built = build_langgraph_agent(
            document_ids_provider=lambda: _current_ids(token),
            document_names_provider=lambda: _current_names(token),
            backend=DEFAULT_BACKEND,
            checkpointer=await _get_checkpointer(),
            gateway_token=token,
        )
        _agent_cache[key] = built
        return built


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "evidenceflow"}


@app.get("/api/session")
async def session_info(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    return _session_status(token) | {"backend": DEFAULT_BACKEND}


@app.get("/api/status")
async def status_info(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    status: dict[str, Any] = {"qdrant_ok": False, "qdrant_error": None}
    try:
        cfg = get_config()
        if not cfg.vector_db.qdrant_url or not cfg.vector_db.qdrant_api_key:
            raise RuntimeError("Qdrant configuration is missing")
        from qdrant_client import QdrantClient

        def check_qdrant() -> bool:
            client = QdrantClient(url=cfg.vector_db.qdrant_url, api_key=cfg.vector_db.qdrant_api_key, timeout=5)
            client.get_collections()
            return True

        status["qdrant_ok"] = await asyncio.to_thread(check_qdrant)
    except Exception as exc:  # noqa: BLE001
        status["qdrant_error"] = str(exc)
    status["web_search_configured"] = bool(os.getenv("TAVILY_API_KEY") or os.getenv("WEB_SEARCH_API_KEY"))
    status["primary_source_configured"] = os.getenv("KATZILLA_ENABLED", "false").lower() in {"1", "true", "yes", "on"} and bool(os.getenv("KATZILLA_API_KEY"))
    return status


@app.get("/api/chats")
async def list_chats(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    return {"chats": _list_chats(token)}


@app.get("/api/chats/{thread_id}")
async def get_chat(thread_id: str, authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    return _load_chat(token, thread_id)


@app.post("/api/chats")
async def create_chat(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    thread_id = uuid.uuid4().hex
    chat = {"thread_id": thread_id, "title": "New research thread", "messages": [], "uploaded_docs": [], "updated_at": None}
    _save_chat(token, chat)
    return chat


@app.delete("/api/chats/{thread_id}")
async def delete_chat(thread_id: str, authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    chat = _load_chat(token, thread_id)
    for doc in chat.get("uploaded_docs") or []:
        try:
            await delete_document(doc["document_id"])
        except Exception:  # noqa: BLE001
            pass
    try:
        built = _agent_cache.get(_session_key(token))
        if built is not None:
            await built["agent"].checkpointer.adelete_thread(thread_id)
    except Exception:  # noqa: BLE001
        pass
    _chat_path(token, thread_id).unlink(missing_ok=True)
    return {"deleted": True}


@app.post("/api/clear")
async def clear_everything(authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    global_docs = _load_global_docs(token)
    chats = _list_chats(token)
    doc_ids = {d["document_id"] for d in global_docs}
    for item in chats:
        chat = _load_chat(token, item["thread_id"])
        doc_ids.update(d["document_id"] for d in chat.get("uploaded_docs") or [])
    for doc_id in doc_ids:
        try:
            await delete_document(doc_id)
        except Exception:  # noqa: BLE001
            pass
    for item in chats:
        _chat_path(token, item["thread_id"]).unlink(missing_ok=True)
        try:
            built = _agent_cache.get(_session_key(token))
            if built is not None:
                await built["agent"].checkpointer.adelete_thread(item["thread_id"])
        except Exception:  # noqa: BLE001
            pass
    _global_docs_path(token).unlink(missing_ok=True)
    return {"cleared": True}


@app.get("/api/documents")
async def list_documents(thread_id: str = "", authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    global_docs, chat_docs = _scope_docs(token, thread_id)
    return {"global": global_docs, "chat": chat_docs, "scope_ids": _scope_ids(token, thread_id)}


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = "",
    scope: str = "chat",
    authorization: str | None = Header(default=None),
):
    token = _token_from_header(authorization)
    _require_active_session(token)
    if scope not in {"global", "chat"}:
        raise HTTPException(status_code=400, detail="scope must be global or chat")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    content_hash = hashlib.sha256(data).hexdigest()
    global_docs, chat_docs = _scope_docs(token, thread_id)
    existing = global_docs if scope == "global" else chat_docs
    if any(d.get("content_hash") == content_hash for d in existing):
        return {"duplicate": True, "document": next(d for d in existing if d.get("content_hash") == content_hash), "result": None}
    result = await asyncio.wait_for(ingest_document(data, source_name=file.filename or "uploaded-document"), timeout=DEFAULT_TURN_TIMEOUT_SECONDS)
    entry = {
        "document_id": result["document_id"],
        "name": file.filename or "uploaded-document",
        "chunk_count": result.get("chunk_count", 0),
        "content_hash": content_hash,
    }
    if scope == "global":
        global_docs.append(entry)
        _save_global_docs(token, global_docs)
    else:
        chat = _load_chat(token, thread_id)
        chat["uploaded_docs"] = [*chat.get("uploaded_docs", []), entry]
        _save_chat(token, chat)
    return {"duplicate": False, "document": entry, "result": _jsonable(result)}


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str, thread_id: str = "", scope: str = "chat", authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    if scope not in {"global", "chat"}:
        raise HTTPException(status_code=400, detail="scope must be global or chat")
    global_docs, chat_docs = _scope_docs(token, thread_id)
    docs = global_docs if scope == "global" else chat_docs
    if not any(d.get("document_id") == document_id for d in docs):
        raise HTTPException(status_code=404, detail="Document not found in this scope")
    await delete_document(document_id)
    if scope == "global":
        _save_global_docs(token, [d for d in global_docs if d.get("document_id") != document_id])
    else:
        chat = _load_chat(token, thread_id)
        chat["uploaded_docs"] = [d for d in chat.get("uploaded_docs", []) if d.get("document_id") != document_id]
        _save_chat(token, chat)
    return {"deleted": True, "document_id": document_id}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, authorization: str | None = Header(default=None)):
    token = _token_from_header(authorization)
    _require_active_session(token)
    query = request.query.strip()
    built = await _ensure_agent(token)
    chat = _load_chat(token, request.thread_id)
    _ensure_title(chat, query)
    chat.setdefault("messages", []).append({"role": "user", "content": query})
    _save_chat(token, chat)

    async def events() -> AsyncIterator[str]:
        ctx = _ACTIVE_THREAD.set(request.thread_id)
        final_payload: dict[str, Any] | None = None
        accumulated = ""
        try:
            yield f"data: {json.dumps({'type': 'status', 'text': 'Starting research…'})}\n\n"
            async for event in astream_langgraph_turn(built, query, thread_id=request.thread_id):
                payload = _jsonable(event)
                if payload.get("type") == "token":
                    accumulated += str(payload.get("text", ""))
                if payload.get("type") == "final":
                    final_payload = payload
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            try:
                _ACTIVE_THREAD.reset(ctx)
            except ValueError:
                pass
            if final_payload:
                meta = _meta_from_final(final_payload)
                answer = _display_answer(final_payload.get("final_answer") or accumulated)
                chat = _load_chat(token, request.thread_id)
                chat["messages"] = chat.get("messages", []) + [{"role": "assistant", "content": answer, "meta": meta}]
                _save_chat(token, chat)
            yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/{path:path}")
async def spa(path: str):
    if path.startswith("api/") or path == "healthz":
        return JSONResponse({"detail": "Not found"}, status_code=404)
    index = WEB_DIST / "index.html"
    return FileResponse(index) if index.exists() else JSONResponse({"detail": "Frontend not built"}, status_code=503)
