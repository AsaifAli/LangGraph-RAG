"""Safe application error logging that never records exception messages."""
from __future__ import annotations

import logging
import uuid


def log_safe_failure(
    logger: logging.Logger,
    event: str,
    error: BaseException,
    *,
    incident_id: str | None = None,
) -> str:
    """Log only an incident reference and exception type; omit raw error text."""
    safe_event = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(event)
    )[:48] or "application"
    reference = incident_id or uuid.uuid4().hex[:12]
    logger.error(
        "%s_failed incident_id=%s error_type=%s",
        safe_event,
        reference,
        type(error).__name__,
    )
    return reference
