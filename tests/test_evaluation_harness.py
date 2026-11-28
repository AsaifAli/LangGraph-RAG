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



def test_evaluation_events_include_version_lineage_but_no_content(monkeypatch):
    from shared import evaluation_harness

    monkeypatch.setenv("EVALUATION_HARNESS_URL", "https://example.invalid")
    monkeypatch.setenv("EVALUATION_HARNESS_INGEST_TOKEN", "test-project-token")
    sent = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, json, headers):
            sent["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(evaluation_harness.httpx, "Client", FakeClient)
    accepted = evaluation_harness.record_event(
        event_type="evaluation.completed",
        status="success",
        workflow="rag_query",
        model="provider/model-a",
        experiment_id="ragas-model-a",
        dataset_version="synthetic-v2",
        prompt_version="synthesis-v4",
        model_version="model-a",
        config_version="retrieval-v3",
        metrics={"external.ragas.faithfulness": 0.93},
    )

    assert accepted is True
    payload = sent["payload"]
    assert payload["event_type"] == "evaluation.completed"
    assert payload["experiment_id"] == "ragas-model-a"
    assert payload["dataset_version"] == "synthetic-v2"
    assert payload["prompt_version"] == "synthesis-v4"
    assert payload["model_version"] == "model-a"
    assert payload["config_version"] == "retrieval-v3"
    assert payload["metrics"] == {"external.ragas.faithfulness": 0.93}
    assert "response" not in payload
    assert "retrieved_contexts" not in payload


def test_runtime_guardrail_metrics_use_only_bounded_numeric_signals():
    from shared.evaluation_harness import build_runtime_evaluation_metrics

    metrics = build_runtime_evaluation_metrics({
        "grounding_status": "PARTIAL",
        "abstained": False,
        "verified_count": 3,
        "proposed_count": 5,
        "quality": {
            "citation_coverage": 80.0,
            "citation_verification": 60.0,
            "numeric_support": {"numeric_claims_supported": True},
        },
        "trust": {
            "label": "REVIEW",
            "security_blocked": False,
        },
        "chunks": [
            {"meta_data": {"graph_expanded": True}},
            {"meta_data": {"graph_expanded": False}},
        ],
        "web_sources": [{"title": "Private Source Title", "url": "https://example.invalid"}],
        "final_answer": "Private generated answer",
        "guardrail_findings": [
            {"kind": "prompt_injection", "message": "PRIVATE FINDING TEXT", "blocked": False},
            {"kind": "secret_leakage", "message": "PRIVATE SECRET FINDING", "blocked": True},
        ],
    })

    assert metrics["runtime.citation_coverage"] == 0.8
    assert metrics["runtime.citation_verification"] == 0.6
    assert metrics["runtime.grounding_verified"] == 0.0
    assert metrics["runtime.trust_verified"] == 0.0
    assert metrics["runtime.trust_review_required"] == 1.0
    assert metrics["runtime.security_blocked"] == 0.0
    assert metrics["runtime.abstained"] == 0.0
    assert metrics["runtime.numeric_claims_supported"] == 1.0
    assert metrics["evidence.verified_count"] == 3
    assert metrics["evidence.proposed_count"] == 5
    assert metrics["retrieval.chunk_count"] == 2
    assert metrics["retrieval.graph_expanded_chunk_count"] == 1
    assert metrics["retrieval.web_source_count"] == 1
    assert metrics["runtime.guardrail_finding_count"] == 2
    assert metrics["runtime.guardrail_blocked"] == 1.0
    assert "PRIVATE FINDING TEXT" not in str(metrics)
    assert "PRIVATE SECRET FINDING" not in str(metrics)
    assert "Private Source Title" not in str(metrics)
    assert "Private generated answer" not in str(metrics)
