from fastapi.testclient import TestClient

from app import api_main
from app.api_store import EvidenceFlowStore


class MemoryKV:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        return "OK"

    def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    def ping(self):
        return True


def _test_store():
    return EvidenceFlowStore(
        kv=MemoryKV(),
        encryption_secret="api-route-integration-secret",
    )


def test_session_status_never_exposes_gateway_bearer_or_owner_scope(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    session_id = test_store.create_session(
        access_token="do-not-return-this-gateway-token",
        owner_scope="a" * 64,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        expires_in=600,
    )

    response = TestClient(api_main.app).get(
        "/api/v1/session",
        cookies={api_main.COOKIE_NAME: session_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["active"] is True
    assert body["project"] == "evidenceflow"
    assert body["provider"] == "openrouter"
    assert body["model"] == "openai/gpt-4o-mini"
    assert "access_token" not in body
    assert "owner_scope" not in body


def test_chat_records_are_inaccessible_across_verified_owner_scopes(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    owner_a_session = test_store.create_session(
        access_token="gateway-token-a",
        owner_scope="a" * 64,
        provider="",
        model="",
        expires_in=600,
    )
    owner_b_session = test_store.create_session(
        access_token="gateway-token-b",
        owner_scope="b" * 64,
        provider="",
        model="",
        expires_in=600,
    )

    client = TestClient(api_main.app)
    created = client.post(
        "/api/v1/chats",
        cookies={api_main.COOKIE_NAME: owner_a_session},
    )
    assert created.status_code == 201
    thread_id = created.json()["thread_id"]

    owner_a_read = client.get(
        f"/api/v1/chats/{thread_id}",
        cookies={api_main.COOKIE_NAME: owner_a_session},
    )
    owner_b_read = client.get(
        f"/api/v1/chats/{thread_id}",
        cookies={api_main.COOKIE_NAME: owner_b_session},
    )

    assert owner_a_read.status_code == 200
    assert owner_b_read.status_code == 404
