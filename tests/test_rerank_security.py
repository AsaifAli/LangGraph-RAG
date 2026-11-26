from __future__ import annotations

import json
import urllib.request

import pytest

import retrieval.rag_pipeline as pipeline


class FakeRerankResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}]
        }).encode("utf-8")


def test_jina_rerank_rejects_non_https_destination(monkeypatch):
    monkeypatch.setattr(pipeline, "JINA_API_KEY", "test-api-key")
    monkeypatch.setattr(pipeline, "JINA_API_URL", "file:///etc/passwd")
    called = False

    def fake_urlopen(*args, **kwargs):
        nonlocal called
        called = True
        return FakeRerankResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    chunk = pipeline.RetrievedChunk(
        evidence_id="E1",
        document_id="doc-1",
        content="Example document",
        score=0.1,
        meta_data={},
    )

    with pytest.raises(ValueError, match="HTTPS"):
        pipeline._jina_rerank("example", [chunk], top_n=1)

    assert called is False
