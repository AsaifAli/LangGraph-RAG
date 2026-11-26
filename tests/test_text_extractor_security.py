import subprocess

import pytest

from shared.exceptions import StorageError
from shared import text_extractors


def test_legacy_doc_conversion_has_a_hard_timeout(monkeypatch):
    class FakeLibConfig:
        libreoffice_path = "/usr/bin/soffice"

    class FakeConfig:
        lib = FakeLibConfig()

    monkeypatch.setattr(text_extractors, "get_config", lambda: FakeConfig())
    observed = {}

    def timeout_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(text_extractors.subprocess, "run", timeout_run)

    with pytest.raises(StorageError, match="timed out"):
        text_extractors.DocumentTextExtractor().doc_bytes_to_text_via_libreoffice(
            b"legacy doc bytes"
        )

    assert observed["timeout"] == 60
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["shell"] is False
