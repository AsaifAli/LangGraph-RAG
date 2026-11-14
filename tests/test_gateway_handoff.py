import json

import pytest

from app import gateway_handoff
from app.gateway_handoff import (
    GatewayHandoffError,
    _exchange_endpoint,
    exchange_handoff_code,
)


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, limit):
        return self.body[:limit]


class FakeOpener:
    def __init__(self, payload, expected_url):
        self.payload = payload
        self.expected_url = expected_url
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout

        assert request.full_url == self.expected_url
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert request.get_header("Cache-control") == "no-store"

        body = json.loads(request.data.decode("utf-8"))
        assert body["project"] == "evidenceflow"
        assert body["code"] == "a" * 43

        return FakeResponse(self.payload)


def test_exchange_endpoint_accepts_gateway_root():
    assert _exchange_endpoint(
        "https://portfolio-llm-gateway.onrender.com"
    ) == (
        "https://portfolio-llm-gateway.onrender.com/"
        "v1/demo/session/exchange"
    )


def test_exchange_endpoint_accepts_v1_base():
    assert _exchange_endpoint(
        "https://portfolio-llm-gateway.onrender.com/v1"
    ) == (
        "https://portfolio-llm-gateway.onrender.com/"
        "v1/demo/session/exchange"
    )


def test_exchange_endpoint_rejects_invalid_urls():
    with pytest.raises(GatewayHandoffError):
        _exchange_endpoint("javascript:alert(1)")

    with pytest.raises(GatewayHandoffError):
        _exchange_endpoint(
            "https://user:password@example.com"
        )


def test_exchange_returns_valid_server_session(monkeypatch):
    expected_url = (
        "https://portfolio-llm-gateway.onrender.com/"
        "v1/demo/session/exchange"
    )

    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "evidenceflow",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        expected_url,
    )

    monkeypatch.setattr(
        gateway_handoff,
        "build_opener",
        lambda *args: fake_opener,
    )

    result = exchange_handoff_code(
        "a" * 43,
        gateway_url="https://portfolio-llm-gateway.onrender.com",
    )

    assert result["access_token"] == "signed-session-token"
    assert result["expires_in"] == 900
    assert result["target_project"] == "evidenceflow"
    assert fake_opener.timeout == 8


def test_exchange_rejects_wrong_target_project(monkeypatch):
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "quotesense",
        },
        "https://gateway.example.com/v1/demo/session/exchange",
    )

    monkeypatch.setattr(
        gateway_handoff,
        "build_opener",
        lambda *args: fake_opener,
    )

    with pytest.raises(
        GatewayHandoffError,
        match="unexpected project",
    ):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://gateway.example.com",
        )


def test_exchange_rejects_missing_expiry(monkeypatch):
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "target_project": "evidenceflow",
        },
        "https://gateway.example.com/v1/demo/session/exchange",
    )

    monkeypatch.setattr(
        gateway_handoff,
        "build_opener",
        lambda *args: fake_opener,
    )

    with pytest.raises(
        GatewayHandoffError,
        match="expiry",
    ):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://gateway.example.com",
        )


def test_exchange_rejects_invalid_code_before_network():
    with pytest.raises(GatewayHandoffError, match="invalid or incomplete"):
        exchange_handoff_code(
            "not-a-valid-code",
            gateway_url="https://gateway.example.com",
        )


def test_exchange_requires_dedicated_gateway_url(monkeypatch):
    monkeypatch.delenv(
        "PORTFOLIO_SESSION_GATEWAY_URL",
        raising=False,
    )

    with pytest.raises(
        GatewayHandoffError,
        match="not configured",
    ):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="",
        )
