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

def test_api_readiness_keeps_render_legacy_probe_compatible_without_restoring_streamlit(monkeypatch):
    class ReadyStore:
        def check_ready(self):
            return None

    monkeypatch.setattr(api_main, "store", ReadyStore())
    monkeypatch.setattr(api_main, "checkpointer", object())

    client = TestClient(api_main.app)
    response = client.get("/ready")
    legacy_health_probe = client.get("/_stcore/health")

    expected = {
        "status": "ready",
        "workspace_store": "ok",
        "checkpointer": "ok",
    }
    assert response.status_code == 200
    assert response.json() == expected
    assert legacy_health_probe.status_code == 200
    assert legacy_health_probe.json() == expected


def test_research_sse_stream_forwards_only_safe_task_lifecycle_events(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    monkeypatch.setattr(api_main.owner_rate_limiter, "enforce", lambda *args, **kwargs: None)
    session_id = test_store.create_session(
        access_token="gateway-child-token-for-task-stream",
        owner_scope="d" * 64,
        provider="openai",
        model="gpt-4o-mini",
        expires_in=600,
    )

    async def fake_build_agent(**kwargs):
        return {"agent": "test-agent"}

    async def fake_stream(agent, message, *, thread_id):
        yield {
            "type": "task",
            "task": {
                "id": "retrieve_kb-1",
                "name": "retrieve_kb",
                "status": "running",
                "prompt": "PRIVATE PROMPT MUST NOT LEAK",
                "internal_reasoning": "PRIVATE CHAIN OF THOUGHT MUST NOT LEAK",
            },
        }
        yield {
            "type": "task",
            "task": {
                "id": "retrieve_kb-1",
                "name": "retrieve_kb",
                "status": "completed",
            },
        }
        yield {
            "type": "final",
            "final_answer": "The policy supports this conclusion.",
            "chunks": [],
            "web_sources": [],
            "delegations": [],
            "used_knowledge_base": True,
            "used_web": False,
            "used_primary_source": False,
            "abstained": False,
            "grounding_status": "VERIFIED",
            "quality": {},
            "trust": {"label": "VERIFIED"},
        }

    monkeypatch.setattr(api_main, "_get_or_build_agent", fake_build_agent)
    monkeypatch.setattr(api_main, "astream_langgraph_turn", fake_stream)

    client = TestClient(api_main.app)
    created = client.post(
        "/api/v1/chats",
        cookies={api_main.COOKIE_NAME: session_id},
    )
    assert created.status_code == 201
    thread_id = created.json()["thread_id"]

    response = client.post(
        f"/api/v1/chats/{thread_id}/messages",
        json={"message": "Summarize the indexed policy."},
        cookies={api_main.COOKIE_NAME: session_id},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: task") == 2
    assert '"label":"Search knowledge base"' in response.text
    assert '"status":"running"' in response.text
    assert '"status":"completed"' in response.text
    assert "PRIVATE PROMPT MUST NOT LEAK" not in response.text
    assert "PRIVATE CHAIN OF THOUGHT MUST NOT LEAK" not in response.text
    assert '"internal_reasoning"' not in response.text


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


def test_chat_scoped_document_listing_only_includes_the_requested_thread(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    owner_scope = "f" * 64
    session_id = test_store.create_session(
        access_token="gateway-token",
        owner_scope=owner_scope,
        provider="",
        model="",
        expires_in=600,
    )
    global_document = {
        "name": "shared.pdf", "document_id": "global-1", "chunk_count": 2,
        "owner_scope": owner_scope, "scope": "global",
    }
    private_a = {
        "name": "thread-a.txt", "document_id": "chat-a", "chunk_count": 1,
        "owner_scope": owner_scope, "scope": "chat",
    }
    private_b = {
        "name": "thread-b.txt", "document_id": "chat-b", "chunk_count": 1,
        "owner_scope": owner_scope, "scope": "chat",
    }
    thread_a = "a" * 32
    thread_b = "b" * 32
    test_store.save_workspace(owner_scope, {
        "global_documents": [global_document],
        "chats": [
            {"thread_id": thread_a, "messages": [], "uploaded_docs": [private_a]},
            {"thread_id": thread_b, "messages": [], "uploaded_docs": [private_b]},
        ],
    })

    client = TestClient(api_main.app)
    headers = {"Cookie": f"{api_main.COOKIE_NAME}={session_id}"}
    response_a = client.get(
        f"/api/v1/documents?scope=all&thread_id={thread_a}", headers=headers
    )
    response_b = client.get(
        f"/api/v1/documents?scope=all&thread_id={thread_b}", headers=headers
    )
    missing_thread = client.get("/api/v1/documents?scope=chat", headers=headers)

    assert response_a.status_code == 200
    names_a = {item["name"] for item in response_a.json()["data"]}
    assert names_a == {"shared.pdf", "thread-a.txt"}
    assert response_b.status_code == 200
    names_b = {item["name"] for item in response_b.json()["data"]}
    assert names_b == {"shared.pdf", "thread-b.txt"}
    assert missing_thread.status_code == 422


def test_chat_scoped_upload_is_indexed_only_for_the_target_conversation(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    owner_scope = "e" * 64
    session_id = test_store.create_session(
        access_token="gateway-token",
        owner_scope=owner_scope,
        provider="",
        model="",
        expires_in=600,
    )
    calls = []

    async def fake_ingest(content, *, source_name, owner_scope):
        calls.append((content, source_name, owner_scope))
        return {
            "document_id": "private-document-1",
            "chunk_count": 3,
            "modalities": ["text"],
            "visual_pages_processed": 0,
            "visual_pages_skipped": 0,
        }

    monkeypatch.setattr(api_main, "ingest_document", fake_ingest)
    client = TestClient(api_main.app)
    cookies = {api_main.COOKIE_NAME: session_id}
    thread_a = client.post("/api/v1/chats", cookies=cookies).json()["thread_id"]
    thread_b = client.post("/api/v1/chats", cookies=cookies).json()["thread_id"]

    uploaded = client.post(
        "/api/v1/documents",
        cookies=cookies,
        data={"scope": "chat", "thread_id": thread_a},
        files={"file": ("thread-private.txt", b"private conversation evidence", "text/plain")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["document"]["scope"] == "chat"
    assert calls == [(b"private conversation evidence", "thread-private.txt", owner_scope)]

    global_library = client.get("/api/v1/documents?scope=global", cookies=cookies)
    thread_a_library = client.get(
        f"/api/v1/documents?scope=all&thread_id={thread_a}", cookies=cookies
    )
    thread_b_library = client.get(
        f"/api/v1/documents?scope=all&thread_id={thread_b}", cookies=cookies
    )

    assert global_library.status_code == 200
    assert global_library.json()["data"] == []
    assert [item["name"] for item in thread_a_library.json()["data"]] == ["thread-private.txt"]
    assert thread_b_library.json()["data"] == []


def test_chat_scoped_upload_rejects_an_unknown_thread_before_indexing(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    session_id = test_store.create_session(
        access_token="gateway-token",
        owner_scope="d" * 64,
        provider="",
        model="",
        expires_in=600,
    )

    async def unexpected_ingest(*args, **kwargs):
        raise AssertionError("Unknown conversations must be rejected before indexing.")

    monkeypatch.setattr(api_main, "ingest_document", unexpected_ingest)
    response = TestClient(api_main.app).post(
        "/api/v1/documents",
        cookies={api_main.COOKIE_NAME: session_id},
        data={"scope": "chat", "thread_id": "c" * 32},
        files={"file": ("private.txt", b"must not be indexed", "text/plain")},
    )

    assert response.status_code == 404


def test_chat_retrieval_scope_contains_global_and_only_active_thread_documents(monkeypatch):
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
    client = TestClient(api_main.app)
    cookies = {api_main.COOKIE_NAME: session_id}
    thread_a = client.post("/api/v1/chats", cookies=cookies).json()["thread_id"]
    thread_b = client.post("/api/v1/chats", cookies=cookies).json()["thread_id"]

    global_document = {
        "name": "shared.pdf", "document_id": "global-doc", "chunk_count": 2,
        "owner_scope": owner_scope, "scope": "global",
    }
    private_a = {
        "name": "thread-a.txt", "document_id": "private-a", "chunk_count": 1,
        "owner_scope": owner_scope, "scope": "chat",
    }
    private_b = {
        "name": "thread-b.txt", "document_id": "private-b", "chunk_count": 1,
        "owner_scope": owner_scope, "scope": "chat",
    }
    state = test_store.get_workspace(owner_scope)
    state["global_documents"] = [global_document]
    for chat in state["chats"]:
        if chat["thread_id"] == thread_a:
            chat["uploaded_docs"] = [private_a]
        elif chat["thread_id"] == thread_b:
            chat["uploaded_docs"] = [private_b]
    test_store.save_workspace(owner_scope, state)

    captured = {}

    def fake_build_agent(**kwargs):
        captured["ids"] = kwargs["document_ids_provider"]()
        captured["names"] = kwargs["document_names_provider"]()
        return {"agent": "test-agent"}

    async def fake_stream(agent, message, *, thread_id):
        yield {
            "type": "final",
            "final_answer": "Evidence from the active conversation.",
            "chunks": [],
            "web_sources": [],
            "delegations": [],
            "used_knowledge_base": True,
            "used_web": False,
            "used_primary_source": False,
            "abstained": False,
            "grounding_status": "VERIFIED",
            "quality": {},
            "trust": {"label": "VERIFIED"},
        }

    monkeypatch.setattr(api_main, "checkpointer", object())
    monkeypatch.setattr(api_main, "build_langgraph_agent", fake_build_agent)
    monkeypatch.setattr(api_main, "astream_langgraph_turn", fake_stream)
    monkeypatch.setattr(api_main.owner_rate_limiter, "enforce", lambda *args, **kwargs: None)

    response = client.post(
        f"/api/v1/chats/{thread_a}/messages",
        json={"message": "What does the private attachment say?"},
        cookies=cookies,
    )

    assert response.status_code == 200
    assert captured["ids"] == ["global-doc", "private-a"]
    assert captured["names"] == {
        "global-doc": "shared.pdf",
        "private-a": "thread-a.txt",
    }
    assert "private-b" not in captured["ids"]



def test_safe_outcome_exposes_bounded_graph_provenance_without_internal_scope():
    outcome = api_main._safe_outcome({
        "final_answer": "Northbridge shares a control with FDA [EVID: E1].",
        "chunks": [{
            "evidence_id": "E1",
            "document_id": "doc-a",
            "content": "Grounded evidence",
            "meta_data": {
                "source_name": "agreement.pdf",
                "graph_expanded": True,
                "graph_hop": 2,
                "graph_shared_entities": ["northbridge", "fda"],
                "owner_scope": "private-owner-scope",
                "entity_terms": ["northbridge", "fda"],
            },
        }],
    })

    metadata = outcome["chunks"][0]["meta_data"]
    assert metadata["graph_expanded"] is True
    assert metadata["graph_hop"] == 2
    assert metadata["graph_shared_entities"] == ["northbridge", "fda"]
    assert "owner_scope" not in metadata
    assert "entity_terms" not in metadata



def test_graph_expansion_metrics_count_internal_retrieved_chunk_dataclasses():
    from retrieval.rag_pipeline import RetrievedChunk

    chunks = [
        RetrievedChunk(
            evidence_id="E1",
            document_id="doc-a",
            content="Graph neighbour",
            score=0.1,
            meta_data={"graph_expanded": True},
        ),
        RetrievedChunk(
            evidence_id="E2",
            document_id="doc-a",
            content="Seed passage",
            score=0.2,
            meta_data={"graph_expanded": False},
        ),
        {
            "evidence_id": "E3",
            "document_id": "doc-a",
            "content": "Another graph neighbour",
            "score": 0.1,
            "meta_data": {"graph_expanded": True},
        },
    ]

    assert api_main._graph_expanded_chunk_count(chunks) == 2


def test_web_citations_link_only_to_sources_returned_by_this_turn():
    outcome = api_main._safe_outcome({
        "final_answer": "The public standard was updated [WEB: W1]. Unsupported [WEB: W9].",
        "web_search_status": "succeeded",
        "source_mode": "web",
        "used_web": True,
        "web_sources": [{
            "evidence_id": "W1",
            "title": "Official standards page",
            "url": "https://example.com/standards",
            "content": "Updated standard excerpt.",
        }],
    })

    assert outcome["final_answer"] == (
        "The public standard was updated [W1](#web-source-W1). Unsupported."
    )
    assert outcome["web_sources"][0]["evidence_id"] == "W1"
    assert outcome["web_sources"][0]["snippet"] == "Updated standard excerpt."
    assert outcome["web_search_status"] == "succeeded"
    assert outcome["source_mode"] == "web"


def test_session_status_reports_web_search_capability_without_exposing_credentials(monkeypatch):
    test_store = _test_store()
    monkeypatch.setattr(api_main, "store", test_store)
    monkeypatch.setattr(
        api_main,
        "get_web_search_capability",
        lambda: {"provider": "Tavily", "status": "not_configured", "available": False},
    )
    session_id = test_store.create_session(
        access_token="secret-gateway-token",
        owner_scope="e" * 64,
        provider="openrouter",
        model="test-model",
        expires_in=600,
    )

    response = TestClient(api_main.app).get(
        "/api/v1/session",
        cookies={api_main.COOKIE_NAME: session_id},
    )

    assert response.status_code == 200
    assert response.json()["capabilities"]["web_search"] == {
        "provider": "Tavily",
        "status": "not_configured",
        "available": False,
    }
    assert "secret-gateway-token" not in response.text
    assert "api_key" not in response.text.lower()
