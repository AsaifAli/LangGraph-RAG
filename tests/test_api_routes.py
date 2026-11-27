from fastapi.testclient import TestClient

from app import api_main, gateway_handoff
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
    assert response.headers["cache-control"] == "no-store"


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



def test_document_api_omits_internal_owner_scope_and_content_hash(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    owner_scope = "c" * 64
    session_id = test_store.create_session(
        access_token="gateway-token",
        owner_scope=owner_scope,
        provider="",
        model="",
        expires_in=600,
    )
    document = {
        "name": "sample.pdf",
        "document_id": "document-123",
        "chunk_count": 8,
        "content_hash": "private-content-fingerprint",
        "owner_scope": owner_scope,
        "scope": "global",
        "created_at": "2026-11-26T00:00:00+00:00",
    }
    chat_document = {
        **document,
        "document_id": "document-chat-456",
        "scope": "chat",
    }
    thread_id = "d" * 32
    test_store.save_workspace(
        owner_scope,
        {
            "global_documents": [document],
            "chats": [{
                "thread_id": thread_id,
                "title": "Private chat",
                "messages": [],
                "uploaded_docs": [chat_document],
                "created_at": "2026-11-26T00:00:00+00:00",
                "updated_at": "2026-11-26T00:00:00+00:00",
            }],
        },
    )

    client = TestClient(api_main.app)
    headers = {"Cookie": f"{api_main.COOKIE_NAME}={session_id}"}
    library_response = client.get("/api/v1/documents?scope=global", headers=headers)
    chat_response = client.get(f"/api/v1/chats/{thread_id}", headers=headers)

    assert library_response.status_code == 200
    assert chat_response.status_code == 200
    public_library_document = library_response.json()["data"][0]
    public_chat_document = chat_response.json()["uploaded_docs"][0]
    for public_document in (public_library_document, public_chat_document):
        assert public_document["document_id"]
        assert "owner_scope" not in public_document
        assert "content_hash" not in public_document


def test_gateway_exchange_sets_httponly_cookie_without_returning_bearer(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    monkeypatch.setattr(
        api_main,
        "exchange_handoff_code",
        lambda code, project, identity_headers=None: {
            "access_token": "gateway-secret-for-browser-session",
            "expires_in": 900,
            "owner_scope": "e" * 64,
            "provider": "openrouter",
            "model": "openai/gpt-4o-mini",
            "target_project": project,
        },
    )

    response = TestClient(api_main.app).post(
        "/api/v1/session/exchange",
        json={"code": "a" * 48},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["active"] is True
    assert "access_token" not in body
    assert "owner_scope" not in body
    cookie_header = response.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header

    session_id = response.cookies.get(api_main.COOKIE_NAME)
    assert session_id
    stored_session = test_store.get_session(session_id)
    assert stored_session is not None
    assert stored_session["access_token"] == "gateway-secret-for-browser-session"


def test_evidence_markers_are_rendered_as_source_anchors():
    assert api_main._strip_evidence_tags(
        "The policy requires approval [EVID:E01]."
    ) == "The policy requires approval [E01](#evidence-E01)."
    assert api_main._strip_evidence_tags(
        "See the control [EVID: E02] for context."
    ) == "See the control [E02](#evidence-E02) for context."

def test_legacy_render_health_path_checks_actual_readiness(monkeypatch):
    class ReadyStore:
        def check_ready(self):
            return None

    monkeypatch.setattr(api_main, "store", ReadyStore())
    monkeypatch.setattr(api_main, "checkpointer", object())

    response = TestClient(api_main.app).get("/_stcore/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "workspace_store": "ok",
        "checkpointer": "ok",
    }


def test_evidenceflow_exchange_forwards_studio_workspace_identity(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    captured = {}

    def fake_exchange(code, *, project, identity_headers=None):
        captured.update({
            "code": code,
            "project": project,
            "identity_headers": identity_headers,
        })
        return {
            "access_token": "gateway-child-token",
            "owner_scope": "a" * 64,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "expires_in": 600,
        }

    monkeypatch.setattr(api_main, "exchange_handoff_code", fake_exchange)
    response = TestClient(api_main.app).post(
        "/api/v1/session/exchange",
        json={"code": "a" * 43},
        headers={
            "X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111",
            "X-Studio-Workspace-Timestamp": "1800000000",
            "X-Studio-Workspace-Signature": "b" * 64,
        },
    )

    assert response.status_code == 200
    assert captured == {
        "code": "a" * 43,
        "project": "evidenceflow",
        "identity_headers": {
            "X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111",
            "X-Studio-Workspace-Timestamp": "1800000000",
            "X-Studio-Workspace-Signature": "b" * 64,
        },
    }


def test_evidenceflow_exchange_rejects_incomplete_workspace_identity(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    monkeypatch.setenv(
        "PORTFOLIO_SESSION_GATEWAY_URL",
        "https://portfolio-llm-gateway.onrender.com",
    )

    def unexpected_opener(*args, **kwargs):
        raise AssertionError("Gateway network must not be reached for an incomplete assertion")

    monkeypatch.setattr(gateway_handoff, "build_opener", unexpected_opener)
    response = TestClient(api_main.app).post(
        "/api/v1/session/exchange",
        json={"code": "a" * 43},
        headers={"X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111"},
    )
    assert response.status_code == 401
    assert "identity assertion" in response.json()["detail"].lower()


def test_evidenceflow_revoke_preserves_local_session_on_gateway_failure_then_retries(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    session_id = test_store.create_session(
        access_token="gateway-child-token",
        owner_scope="a" * 64,
        provider="openai",
        model="gpt-4o-mini",
        expires_in=600,
    )
    calls = []

    def fail_revoke(token):
        calls.append(token)
        raise api_main.GatewayHandoffError("Gateway revocation could not be confirmed.")

    monkeypatch.setattr(api_main, "revoke_gateway_session", fail_revoke)
    client = TestClient(api_main.app)
    failed = client.post(
        "/api/v1/session/revoke",
        cookies={api_main.COOKIE_NAME: session_id},
    )
    assert failed.status_code == 503
    assert "set-cookie" not in failed.headers
    assert test_store.get_session(session_id)["access_token"] == "gateway-child-token"

    def confirm_revoke(token):
        calls.append(token)
        return True

    monkeypatch.setattr(api_main, "revoke_gateway_session", confirm_revoke)
    succeeded = client.post(
        "/api/v1/session/revoke",
        cookies={api_main.COOKIE_NAME: session_id},
    )
    assert succeeded.status_code == 200
    assert succeeded.json() == {"revoked": True}
    assert "max-age=0" in succeeded.headers.get("set-cookie", "").lower()
    assert test_store.get_session(session_id) is None
    assert calls == ["gateway-child-token", "gateway-child-token"]
