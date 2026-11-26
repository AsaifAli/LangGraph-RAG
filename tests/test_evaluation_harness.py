import json

class FakeResponse:
    status_code = 202


def test_telemetry_is_a_noop_without_harness_configuration(monkeypatch):
    from shared.evaluation_harness import record_event
    monkeypatch.delenv("EVALUATION_HARNESS_URL", raising=False)
    monkeypatch.delenv("EVALUATION_HARNESS_INGEST_TOKEN", raising=False)
    assert record_event(event_type="run.completed", status="success", workflow="rag_query") is False


def test_telemetry_sends_only_allowlisted_metadata(monkeypatch):
    from shared import evaluation_harness

    monkeypatch.setenv("EVALUATION_HARNESS_URL", "https://example.invalid")
    monkeypatch.setenv("EVALUATION_HARNESS_INGEST_TOKEN", "test-project-token")
    sent = {}

    class FakeClient:
        def __init__(self, **kwargs):
            sent["client_options"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, json, headers):
            sent["url"] = url
            sent["payload"] = json
            sent["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(evaluation_harness.httpx, "Client", FakeClient)
    result = evaluation_harness.record_event(
        event_type="run.completed",
        status="warning",
        workflow="rag_query",
        duration_ms=12.5,
        model="test-model",
        metrics={
            "retrieved_chunks": 4,
            "evidence_audit_pass": False,
            "prompt_text": "must not be included",
        },
    )

    assert result is True
    assert sent["client_options"]["timeout"] == 1.25
    assert sent["client_options"]["follow_redirects"] is False
    assert sent["client_options"]["trust_env"] is False
    assert sent["payload"]["project"] == "evidenceflow"
    assert sent["payload"]["metrics"] == {
        "retrieved_chunks": 4,
        "evidence_audit_pass": False,
    }
    assert "prompt_text" not in sent["payload"]["metrics"]
    assert "Authorization" not in sent["payload"]
    assert sent["headers"]["Authorization"] == "Bearer test-project-token"


def test_telemetry_rejects_non_https_destination(monkeypatch):
    from shared import evaluation_harness

    monkeypatch.setenv("EVALUATION_HARNESS_URL", "file:///etc")
    monkeypatch.setenv("EVALUATION_HARNESS_INGEST_TOKEN", "test-project-token")
    called = False

    class FakeClient:
        def __init__(self, **kwargs):
            nonlocal called
            called = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(evaluation_harness.httpx, "Client", FakeClient)

    assert evaluation_harness.record_event(
        event_type="run.completed",
        status="success",
        workflow="rag_query",
    ) is False
    assert called is False
