"""Best-effort, content-free telemetry to the studio evaluation service."""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from urllib.parse import urlsplit

import httpx

LOGGER = logging.getLogger(__name__)


def record_event(
    *,
    event_type: str,
    status: str,
    workflow: str,
    duration_ms: float | None = None,
    model: str | None = None,
    metrics: dict | None = None,
    request_id: str | None = None,
) -> bool:
    """Send bounded event metadata. Missing config/network must never fail the app."""
    base_url = os.getenv("EVALUATION_HARNESS_URL", "").strip().rstrip("/")
    token = os.getenv("EVALUATION_HARNESS_INGEST_TOKEN", "").strip()
    if not base_url or not token:
        return False

    safe_metrics = {}
    for key, value in (metrics or {}).items():
        if (
            isinstance(key, str)
            and key
            and len(key) <= 48
            and all(ch.isalnum() or ch in "_@.-" for ch in key)
            and isinstance(value, (bool, int, float))
            and (not isinstance(value, float) or math.isfinite(value))
        ):
            safe_metrics[key] = value
        if len(safe_metrics) >= 24:
            break

    payload = {
        "event_id": uuid.uuid4().hex,
        "project": "evidenceflow",
        "event_type": event_type,
        "status": status,
        "workflow": str(workflow)[:80],
        "duration_ms": max(0, min(900000, float(duration_ms))) if duration_ms is not None else None,
        "model": str(model)[:120] if model else None,
        "request_id": str(request_id)[:100] if request_id else None,
        "metrics": safe_metrics,
    }
    try:
        parsed = urlsplit(base_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except ValueError:
        return False

    local_http = scheme == "http" and hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not hostname
        or (scheme != "https" and not local_http)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port not in ({None, 443} if scheme == "https" else {None, 80})
    ):
        return False

    try:
        with httpx.Client(
            timeout=1.25,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.post(
                f"{base_url}/api/application-events",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return 200 <= int(response.status_code) < 300
    except (httpx.HTTPError, TimeoutError, OSError, ValueError):
        LOGGER.debug("Evaluation telemetry delivery failed; continuing without it.")
        return False
    except Exception:
        LOGGER.debug("Evaluation telemetry failed unexpectedly; continuing without it.")
        return False
