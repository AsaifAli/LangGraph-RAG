import json

class FakeResponse:
    status = 202
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


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

    def fake_urlopen(request, timeout):
        sent["payload"] = json.loads(request.data.decode("utf-8"))
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(evaluation_harness, "urlopen", fake_urlopen)
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
    assert sent["timeout"] == 1.25
    assert sent["payload"]["project"] == "evidenceflow"
    assert sent["payload"]["metrics"] == {
        "retrieved_chunks": 4,
        "evidence_audit_pass": False,
    }
    assert "prompt_text" not in sent["payload"]["metrics"]
    assert "Authorization" not in sent["payload"]
