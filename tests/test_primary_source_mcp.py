import os

from tools.primary_source_mcp import KatzillaMCPClient


def test_katzilla_disabled_without_flag_or_key(monkeypatch):
    monkeypatch.delenv("KATZILLA_ENABLED", raising=False)
    monkeypatch.delenv("KATZILLA_API_KEY", raising=False)
    client = KatzillaMCPClient.from_env()
    assert not client.available


def test_katzilla_enabled_with_key(monkeypatch):
    monkeypatch.setenv("KATZILLA_ENABLED", "true")
    monkeypatch.setenv("KATZILLA_API_KEY", "kz_test")
    client = KatzillaMCPClient.from_env()
    assert client.available
    assert client.url.endswith("/mcp")
