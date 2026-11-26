from __future__ import annotations

import json

import pytest

import retrieval.rag_pipeline as pipeline


class FakeRerankResponse:
    status_code = 200
    headers = {}

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield self.body


def make_chunk():
    return pipeline.RetrievedChunk(
        evidence_id="E1",
        document_id="doc-1",
        content="Example document",
        score=0.1,
        meta_data={},
    )


def test_jina_rerank_rejects_non_https_destination(monkeypatch):
    monkeypatch.setattr(pipeline, "JINA_API_KEY", "test-api-key")
    monkeypatch.setattr(pipeline, "JINA_API_URL", "file:///etc/passwd")
    called = False

    class FakeClient:
        def __init__(self, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)

    with pytest.raises(ValueError, match="HTTPS"):
        pipeline._jina_rerank("example", [make_chunk()], top_n=1)

    assert called is False


def test_jina_rerank_disables_redirects_and_bounds_response(monkeypatch):
    monkeypatch.setattr(pipeline, "JINA_API_KEY", "test-api-key")
    monkeypatch.setattr(pipeline, "JINA_API_URL", "https://api.jina.ai/v1/rerank")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["options"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, *, json, headers):
            captured.update({
                "method": method,
                "url": url,
                "payload": json,
                "headers": headers,
            })
            return FakeRerankResponse(
                b'{"results":[{"index":0,"relevance_score":0.9}]}'
            )

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    chunk = make_chunk()

    result = pipeline._jina_rerank("example", [chunk], top_n=1)

    assert result == [chunk]
    assert chunk.score == 0.9
    assert captured["options"]["follow_redirects"] is False
    assert captured["options"]["trust_env"] is False
    assert captured["url"] == "https://api.jina.ai/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer test-api-key"


def test_jina_rerank_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr(pipeline, "JINA_API_KEY", "test-api-key")
    monkeypatch.setattr(pipeline, "JINA_API_URL", "https://api.jina.ai/v1/rerank")
    response_body = b"x" * (2_000_001)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeRerankResponse(response_body)

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="size limit"):
        pipeline._jina_rerank("example", [make_chunk()], top_n=1)
