import asyncio
import sys
import types
from types import SimpleNamespace


def _install_fake_ragas(monkeypatch):
    openai = types.ModuleType("openai")
    client_options = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            client_options.update(kwargs)

        async def close(self):
            client_options["closed"] = True

    openai.AsyncOpenAI = FakeAsyncOpenAI

    ragas = types.ModuleType("ragas")
    ragas.__path__ = []
    llms = types.ModuleType("ragas.llms")

    def llm_factory(model, *, client):
        client_options["model"] = model
        client_options["client"] = client
        return object()

    llms.llm_factory = llm_factory

    collections = types.ModuleType("ragas.metrics.collections")

    class FakeMetric:
        def __init__(self, *, llm):
            self.llm = llm

    class Faithfulness(FakeMetric):
        async def ascore(self, **sample):
            assert sample["user_input"] == "What is the policy period?"
            assert sample["response"] == "The policy period is seven years."
            assert len(sample["retrieved_contexts"]) == 2
            return SimpleNamespace(value=0.93, reason="Claims are supported.")

    class ContextPrecisionWithoutReference(FakeMetric):
        async def ascore(self, **sample):
            return SimpleNamespace(value=0.88, reason="Retrieved contexts are relevant.")

    collections.Faithfulness = Faithfulness
    collections.ContextPrecisionWithoutReference = ContextPrecisionWithoutReference

    metrics_package = types.ModuleType("ragas.metrics")
    metrics_package.__path__ = []
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_package)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)
    return client_options


def test_opt_in_ragas_runtime_uses_gateway_and_reports_scores_only(monkeypatch):
    from shared import ragas_evaluation

    client_options = _install_fake_ragas(monkeypatch)
    reported = {}
    monkeypatch.setattr(ragas_evaluation, "record_event", lambda **kwargs: reported.update(kwargs) or True)

    result = asyncio.run(ragas_evaluation.evaluate_rag_with_ragas(
        user_input="What is the policy period?",
        response="The policy period is seven years.",
        retrieved_contexts=[
            "The retention policy is seven years.",
            "Records are kept for seven years from closure.",
        ],
        gateway_token="short-lived-gateway-session",
        gateway_url="https://gateway.example/v1",
        model_label="provider/model-a",
        experiment_id="evidenceflow-live-ragas-a",
        dataset_version="live-rag-v1",
        prompt_version="synthesis-v4",
        config_version="retrieval-v3",
    ))

    assert result == {
        "external.ragas.faithfulness": 0.93,
        "external.ragas.context_precision": 0.88,
    }
    assert client_options["api_key"] == "short-lived-gateway-session"
    assert client_options["base_url"] == "https://gateway.example/v1"
    assert client_options["model"] == "session-model"
    assert client_options["closed"] is True
    assert reported["event_type"] == "evaluation.completed"
    assert reported["model"] == "provider/model-a"
    assert reported["experiment_id"] == "evidenceflow-live-ragas-a"
    assert reported["dataset_version"] == "live-rag-v1"
    assert reported["prompt_version"] == "synthesis-v4"
    assert reported["config_version"] == "retrieval-v3"
    assert reported["metrics"] == result
    assert "user_input" not in str(reported)
    assert "response" not in str(reported)
    assert "retrieved_contexts" not in str(reported)
    assert "short-lived-gateway-session" not in str(reported)


def test_ragas_runtime_rejects_untrusted_gateway_url(monkeypatch):
    from shared.ragas_evaluation import evaluate_rag_with_ragas

    result = asyncio.run(evaluate_rag_with_ragas(
        user_input="question",
        response="answer",
        retrieved_contexts=["context"],
        gateway_token="test-token",
        gateway_url="http://example.com/v1",
        model_label="provider/model",
    ))
    assert result == {}


def test_ragas_runtime_is_opt_in_and_disabled_by_default(monkeypatch):
    from shared.ragas_evaluation import ragas_evaluation_enabled

    monkeypatch.delenv("EVALUATION_RAGAS_ENABLED", raising=False)
    assert ragas_evaluation_enabled() is False
    monkeypatch.setenv("EVALUATION_RAGAS_ENABLED", "true")
    assert ragas_evaluation_enabled() is True
