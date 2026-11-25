import os

from tools.primary_source_mcp import KatzillaMCPClient


def test_katzilla_disabled_without_flag_or_key(monkeypatch):
    monkeypatch.delenv("KATZILLA_ENABLED", raising=False)
    monkeypatch.delenv("KATZILLA_API_KEY", raising=False)
    client = KatzillaMCPClient.from_env()
    assert not client.available


def test_katzilla_enabled_with_key(monkeypatch):
    monkeypatch.setenv("KATZILLA_MCP_ALLOWED_HOSTS", "api.katzilla.dev")
    monkeypatch.setenv("KATZILLA_ENABLED", "true")
    monkeypatch.setenv("KATZILLA_API_KEY", "kz_test")
    client = KatzillaMCPClient.from_env()
    assert client.available
    assert client.url.endswith("/mcp")


def test_mcp_transport_does_not_forward_credentials_through_redirects():
    timeout = object()
    options = KatzillaMCPClient._http_client_options("server-secret", timeout)

    assert options["follow_redirects"] is False
    assert options["trust_env"] is False
    assert options["timeout"] is timeout
    assert options["headers"] == {"Authorization": "Bearer server-secret"}



def test_katzilla_destination_rejects_unapproved_hosts_and_private_dns(monkeypatch):
    import socket
    import pytest

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("KATZILLA_MCP_ALLOWED_HOSTS", "api.katzilla.dev")
    with pytest.raises(ValueError, match="not approved"):
        KatzillaMCPClient._validate_destination("https://attacker.example/mcp")

    def private_dns(*args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("10.0.0.10", 443),
            )
        ]

    monkeypatch.setattr("tools.primary_source_mcp.socket.getaddrinfo", private_dns)
    with pytest.raises(ValueError, match="public addresses"):
        KatzillaMCPClient._validate_destination("https://api.katzilla.dev/mcp")


def test_katzilla_destination_rejects_cleartext_and_credentials(monkeypatch):
    import pytest

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("KATZILLA_MCP_ALLOWED_HOSTS", "api.katzilla.dev")
    with pytest.raises(ValueError, match="HTTPS"):
        KatzillaMCPClient._validate_destination("http://api.katzilla.dev/mcp")
    with pytest.raises(ValueError, match="invalid"):
        KatzillaMCPClient._validate_destination("https://user:secret@api.katzilla.dev/mcp")
