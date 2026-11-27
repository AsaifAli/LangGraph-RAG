import json
import socket

import pytest

from app import gateway_handoff
from app.gateway_handoff import (
    GatewayHandoffError,
    _exchange_endpoint,
    exchange_handoff_code,
)



@pytest.fixture(autouse=True)
def _public_dns_for_gateway_tests(monkeypatch):
    monkeypatch.setenv(
        "PORTFOLIO_SESSION_GATEWAY_ALLOWED_HOSTS",
        "portfolio-llm-gateway.onrender.com,gateway.example.com",
    )

    def resolve_public(*args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            )
        ]

    monkeypatch.setattr(gateway_handoff.socket, "getaddrinfo", resolve_public)


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
            "owner_scope": "a" * 64,
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
    assert result["owner_scope"] == "a" * 64
    assert fake_opener.timeout == 8


def test_exchange_rejects_wrong_target_project(monkeypatch):
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "quotesense",
            "owner_scope": "a" * 64,
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
            "owner_scope": "a" * 64,
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


def test_exchange_rejects_remote_http_before_network():
    with pytest.raises(GatewayHandoffError, match="HTTPS"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="http://gateway.example.com",
        )


def test_exchange_rejects_private_destination_dns(monkeypatch):
    def resolve_private(*args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("169.254.169.254", 443),
            )
        ]

    monkeypatch.setattr(gateway_handoff.socket, "getaddrinfo", resolve_private)
    with pytest.raises(GatewayHandoffError, match="public"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://gateway.example.com",
        )


def test_exchange_keeps_local_http_development_support(monkeypatch):
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "evidenceflow",
            "owner_scope": "a" * 64,
        },
        "http://127.0.0.1:8000/v1/demo/session/exchange",
    )
    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: fake_opener)
    result = exchange_handoff_code(
        "a" * 43,
        gateway_url="http://127.0.0.1:8000",
    )
    assert result["target_project"] == "evidenceflow"
    assert result["owner_scope"] == "a" * 64



def test_exchange_rejects_missing_or_invalid_owner_scope(monkeypatch):
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "evidenceflow",
            "owner_scope": "not-an-owner-hash",
        },
        "https://gateway.example.com/v1/demo/session/exchange",
    )
    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: fake_opener)
    with pytest.raises(GatewayHandoffError, match="access scope"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://gateway.example.com",
        )



def test_gateway_handoff_rejects_unapproved_public_host(monkeypatch):
    monkeypatch.setenv(
        "PORTFOLIO_SESSION_GATEWAY_ALLOWED_HOSTS",
        "portfolio-llm-gateway.onrender.com",
    )
    with pytest.raises(GatewayHandoffError, match="not approved"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://attacker.example.com",
        )


def test_exchange_forwards_only_valid_studio_workspace_identity_headers(monkeypatch):
    expected_url = "https://portfolio-llm-gateway.onrender.com/v1/demo/session/exchange"
    fake_opener = FakeOpener(
        {
            "access_token": "signed-session-token",
            "expires_in": 900,
            "target_project": "evidenceflow",
            "owner_scope": "a" * 64,
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        expected_url,
    )
    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: fake_opener)
    identity = {
        "X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111",
        "X-Studio-Workspace-Timestamp": "1800000000",
        "X-Studio-Workspace-Signature": "b" * 64,
    }

    exchange_handoff_code(
        "a" * 43,
        gateway_url="https://portfolio-llm-gateway.onrender.com",
        identity_headers=identity,
    )

    assert fake_opener.request.get_header("X-studio-workspace-client-id") == identity["X-Studio-Workspace-Client-Id"]
    assert fake_opener.request.get_header("X-studio-workspace-timestamp") == identity["X-Studio-Workspace-Timestamp"]
    assert fake_opener.request.get_header("X-studio-workspace-signature") == identity["X-Studio-Workspace-Signature"]


def test_exchange_rejects_partial_or_malformed_workspace_identity_before_network(monkeypatch):
    called = False

    def unexpected_opener(*args):
        nonlocal called
        called = True
        raise AssertionError("Gateway must not be called with an invalid workspace assertion")

    monkeypatch.setattr(gateway_handoff, "build_opener", unexpected_opener)
    with pytest.raises(GatewayHandoffError, match="identity assertion"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://portfolio-llm-gateway.onrender.com",
            identity_headers={"X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111"},
        )
    with pytest.raises(GatewayHandoffError, match="identity assertion"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://portfolio-llm-gateway.onrender.com",
            identity_headers={
                "X-Studio-Workspace-Client-Id": "not-a-uuid",
                "X-Studio-Workspace-Timestamp": "1800000000",
                "X-Studio-Workspace-Signature": "b" * 64,
            },
        )
    assert called is False


def test_revoke_gateway_session_uses_child_bearer_and_requires_confirmation(monkeypatch):
    class RevokeOpener:
        def __init__(self):
            self.request = None
            self.timeout = None

        def open(self, request, timeout):
            self.request = request
            self.timeout = timeout
            assert request.full_url == "https://portfolio-llm-gateway.onrender.com/v1/demo/session/revoke"
            assert request.get_method() == "POST"
            assert request.get_header("Authorization") == "Bearer child-gateway-token"
            assert request.get_header("Cache-control") == "no-store"
            return FakeResponse({"revoked": True})

    opener = RevokeOpener()
    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: opener)

    assert gateway_handoff.revoke_gateway_session(
        "child-gateway-token",
        gateway_url="https://portfolio-llm-gateway.onrender.com",
    ) is True
    assert opener.timeout == 8


def test_revoke_gateway_session_treats_gateway_401_as_already_inactive(monkeypatch):
    from urllib.error import HTTPError
    from io import BytesIO

    class RevokeOpener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 401, "inactive", {}, BytesIO(b"{}"))

    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: RevokeOpener())
    assert gateway_handoff.revoke_gateway_session(
        "expired-child-token",
        gateway_url="https://portfolio-llm-gateway.onrender.com",
    ) is True


def test_revoke_gateway_session_fails_closed_without_confirmation(monkeypatch):
    class RevokeOpener:
        def open(self, request, timeout):
            return FakeResponse({"revoked": False})

    monkeypatch.setattr(gateway_handoff, "build_opener", lambda *args: RevokeOpener())
    with pytest.raises(GatewayHandoffError, match="confirm"):
        gateway_handoff.revoke_gateway_session(
            "child-gateway-token",
            gateway_url="https://portfolio-llm-gateway.onrender.com",
        )


def test_exchange_rejects_timestamp_shape_gateway_will_not_accept(monkeypatch):
    def unexpected_opener(*args):
        raise AssertionError("Gateway must not be called with an invalid timestamp")

    monkeypatch.setattr(gateway_handoff, "build_opener", unexpected_opener)
    with pytest.raises(GatewayHandoffError, match="identity assertion"):
        exchange_handoff_code(
            "a" * 43,
            gateway_url="https://portfolio-llm-gateway.onrender.com",
            identity_headers={
                "X-Studio-Workspace-Client-Id": "11111111-1111-4111-8111-111111111111",
                "X-Studio-Workspace-Timestamp": "1800000000000",
                "X-Studio-Workspace-Signature": "b" * 64,
            },
        )
