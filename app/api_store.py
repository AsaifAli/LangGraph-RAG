from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

_OWNER_SCOPE = re.compile(r"^[a-f0-9]{64}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{40,100}$")
_SESSION_PREFIX = "evidenceflow:api:v1:session:"
_WORKSPACE_PREFIX = "evidenceflow:api:v1:workspace:"
_WORKSPACE_TTL_SECONDS = 90 * 24 * 60 * 60


class SessionStoreUnavailable(RuntimeError):
    """Raised when the shared production session store is unavailable."""


class _LocalKV:
    """Small TTL-aware development store; never selected in production."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at <= time.time():
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ex: int | None = None) -> str:
        expires_at = time.time() + ex if ex else None
        with self._lock:
            self._values[key] = (value, expires_at)
        return "OK"

    def delete(self, key: str) -> int:
        with self._lock:
            return int(self._values.pop(key, None) is not None)

    def ping(self) -> bool:
        return True


_LOCAL_KV = _LocalKV()


def _is_production() -> bool:
    return (
        os.getenv("APP_ENV", "").strip().lower() in {"prod", "production"}
        or os.getenv("RENDER", "").strip().lower() in {"1", "true"}
        or bool(os.getenv("K_SERVICE"))
    )


def _shared_kv():
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if not url or not token:
        if _is_production():
            raise SessionStoreUnavailable(
                "Shared session storage is not configured."
            )
        return _LOCAL_KV
    try:
        from upstash_redis import Redis

        return Redis(
            url=url,
            token=token,
            rest_retries=0,
            allow_telemetry=False,
        )
    except Exception as exc:
        raise SessionStoreUnavailable(
            "Shared session storage could not be initialized."
        ) from exc


class EvidenceFlowStore:
    """Encrypted short-lived sessions and owner-scoped workspace metadata.

    Gateway bearer tokens are encrypted before leaving this process. Redis
    stores only opaque session records and bounded workspace metadata; user
    documents and source content remain in the existing OpenSearch service.
    """

    def __init__(
        self,
        *,
        kv: Any | None = None,
        encryption_secret: str | None = None,
    ) -> None:
        self._kv = kv
        secret = (
            encryption_secret
            or os.getenv("EVIDENCEFLOW_SESSION_ENCRYPTION_KEY", "").strip()
            or os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
        )
        if not secret:
            if _is_production():
                raise SessionStoreUnavailable(
                    "Session encryption is not configured."
                )
            secret = "evidenceflow-local-development-only"
        key = base64.urlsafe_b64encode(
            hashlib.sha256(secret.encode("utf-8")).digest()
        )
        self._fernet = Fernet(key)

    @property
    def kv(self):
        if self._kv is None:
            self._kv = _shared_kv()
        return self._kv

    @staticmethod
    def _owner_key(owner_scope: str) -> str:
        scope = str(owner_scope or "").strip().lower()
        if not _OWNER_SCOPE.fullmatch(scope):
            raise ValueError("A verified owner scope is required.")
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"{_WORKSPACE_PREFIX}{digest}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        candidate = str(session_id or "").strip()
        if not _SESSION_ID.fullmatch(candidate):
            raise ValueError("Invalid application session.")
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return f"{_SESSION_PREFIX}{digest}"

    def create_session(
        self,
        *,
        access_token: str,
        owner_scope: str,
        provider: str,
        model: str,
        expires_in: int,
    ) -> str:
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("A Gateway session token is required.")
        if not _OWNER_SCOPE.fullmatch(str(owner_scope or "").strip().lower()):
            raise ValueError("A verified owner scope is required.")
        if isinstance(expires_in, bool) or not isinstance(expires_in, int):
            raise ValueError("Session lifetime must be an integer.")
        if not 1 <= expires_in <= 3600:
            raise ValueError("Session lifetime is outside the supported range.")

        session_id = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + expires_in
        encrypted_token = self._fernet.encrypt(
            access_token.encode("utf-8")
        ).decode("ascii")
        record = {
            "gateway_token": encrypted_token,
            "owner_scope": owner_scope,
            "provider": str(provider or "")[:80],
            "model": str(model or "")[:160],
            "expires_at": expires_at,
        }
        self.kv.set(
            self._session_key(session_id),
            json.dumps(record, separators=(",", ":")),
            ex=expires_in,
        )
        return session_id

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        try:
            key = self._session_key(session_id)
            raw = self.kv.get(key)
        except ValueError:
            return None
        except SessionStoreUnavailable:
            raise
        except Exception as exc:
            raise SessionStoreUnavailable(
                "Shared session storage is temporarily unavailable."
            ) from exc
        if not raw:
            return None
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("Invalid session record.")
            owner_scope = str(record.get("owner_scope") or "")
            if not _OWNER_SCOPE.fullmatch(owner_scope):
                raise ValueError("Invalid session owner.")
            expires_at = int(record.get("expires_at") or 0)
            if expires_at <= int(time.time()):
                self.revoke_session(session_id)
                return None
            encrypted = record.get("gateway_token")
            if not isinstance(encrypted, str):
                raise ValueError("Invalid encrypted session.")
            access_token = self._fernet.decrypt(
                encrypted.encode("ascii")
            ).decode("utf-8")
            return {
                "access_token": access_token,
                "owner_scope": owner_scope,
                "provider": str(record.get("provider") or ""),
                "model": str(record.get("model") or ""),
                "expires_at": expires_at,
            }
        except (InvalidToken, UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
            self.revoke_session(session_id)
            return None

    def revoke_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        try:
            self.kv.delete(self._session_key(session_id))
        except ValueError:
            return
        except Exception as exc:
            raise SessionStoreUnavailable(
                "Shared session storage is temporarily unavailable."
            ) from exc

    def get_workspace(self, owner_scope: str) -> dict[str, Any]:
        key = self._owner_key(owner_scope)
        try:
            raw = self.kv.get(key)
        except Exception as exc:
            if isinstance(exc, (ValueError, SessionStoreUnavailable)):
                raise
            raise SessionStoreUnavailable(
                "Workspace storage is temporarily unavailable."
            ) from exc
        if not raw:
            return {"global_documents": [], "chats": []}
        try:
            state = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {"global_documents": [], "chats": []}
        if not isinstance(state, dict):
            return {"global_documents": [], "chats": []}
        documents = state.get("global_documents", [])
        chats = state.get("chats", [])
        return {
            "global_documents": documents if isinstance(documents, list) else [],
            "chats": chats if isinstance(chats, list) else [],
        }

    def save_workspace(self, owner_scope: str, state: dict[str, Any]) -> None:
        key = self._owner_key(owner_scope)
        if not isinstance(state, dict):
            raise ValueError("Workspace state must be an object.")
        documents = state.get("global_documents", [])
        chats = state.get("chats", [])
        if not isinstance(documents, list) or not isinstance(chats, list):
            raise ValueError("Workspace collections must be arrays.")
        bounded_state = {
            "global_documents": documents,
            "chats": chats[-100:],
        }
        try:
            self.kv.set(
                key,
                json.dumps(bounded_state, separators=(",", ":")),
                ex=_WORKSPACE_TTL_SECONDS,
            )
        except Exception as exc:
            raise SessionStoreUnavailable(
                "Workspace storage is temporarily unavailable."
            ) from exc

    def check_ready(self) -> bool:
        try:
            return bool(self.kv.ping())
        except Exception as exc:
            raise SessionStoreUnavailable(
                "Shared session storage is not ready."
            ) from exc
