import logging

from shared.security_logging import log_safe_failure


def test_safe_failure_log_omits_exception_message_and_returns_reference(caplog):
    secret = "provider-token-that-must-not-be-logged"
    error = RuntimeError(f"request failed with {secret}")

    with caplog.at_level(logging.ERROR):
        incident_id = log_safe_failure(
            logging.getLogger("security-test"),
            "document_ingest",
            error,
        )

    assert len(incident_id) == 12
    assert "document_ingest_failed" in caplog.text
    assert incident_id in caplog.text
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text
    assert str(error) not in caplog.text


def test_safe_failure_log_accepts_reproducible_incident_reference(caplog):
    with caplog.at_level(logging.ERROR):
        result = log_safe_failure(
            logging.getLogger("security-test"),
            "chat_turn",
            ValueError("sensitive raw provider response"),
            incident_id="abc123def456",
        )

    assert result == "abc123def456"
    assert "abc123def456" in caplog.text
    assert "sensitive raw provider response" not in caplog.text
