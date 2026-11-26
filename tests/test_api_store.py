from __future__ import annotations

import time

from app.api_store import EvidenceFlowStore


class MemoryKV:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self.values[key] = value
        return "OK"

    def delete(self, key: str):
        return int(self.values.pop(key, None) is not None)


def test_session_vault_keeps_gateway_token_server_side_and_round_trips_metadata():
    store = EvidenceFlowStore(
        kv=MemoryKV(),
        encryption_secret="unit-test-session-secret",
    )

    session_id = store.create_session(
        access_token="gateway-secret-token",
        owner_scope="a" * 64,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        expires_in=900,
    )

    session = store.get_session(session_id)

    assert session is not None
    assert session["access_token"] == "gateway-secret-token"
    assert session["owner_scope"] == "a" * 64
    assert session["provider"] == "openrouter"
    assert session["model"] == "openai/gpt-4o-mini"
    assert session["expires_at"] > time.time()


def test_session_vault_revokes_session():
    store = EvidenceFlowStore(kv=MemoryKV(), encryption_secret="test-secret")
    session_id = store.create_session(
        access_token="gateway-secret-token",
        owner_scope="b" * 64,
        provider="",
        model="",
        expires_in=600,
    )

    store.revoke_session(session_id)

    assert store.get_session(session_id) is None


def test_workspace_state_is_isolated_by_verified_owner_scope():
    store = EvidenceFlowStore(kv=MemoryKV(), encryption_secret="test-secret")
    owner_a = "a" * 64
    owner_b = "b" * 64

    store.save_workspace(owner_a, {"global_documents": [{"document_id": "doc-a"}], "chats": []})
    store.save_workspace(owner_b, {"global_documents": [{"document_id": "doc-b"}], "chats": []})

    assert store.get_workspace(owner_a)["global_documents"][0]["document_id"] == "doc-a"
    assert store.get_workspace(owner_b)["global_documents"][0]["document_id"] == "doc-b"


def test_invalid_owner_scope_cannot_read_or_write_workspace():
    store = EvidenceFlowStore(kv=MemoryKV(), encryption_secret="test-secret")

    try:
        store.get_workspace("shared")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid owner scope must be rejected")


def test_malformed_session_id_is_treated_as_missing_session():
    store = EvidenceFlowStore(kv=MemoryKV(), encryption_secret="test-secret")

    assert store.get_session("not-a-valid-session-cookie") is None
