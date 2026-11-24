"""Best-effort, content-free telemetry to the studio evaluation service."""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from urllib.error import URLError
from urllib.request import Request, urlopen

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
    request = Request(
        f"{base_url}/api/application-events",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=1.25) as response:
            return 200 <= int(response.status) < 300
    except (URLError, TimeoutError, OSError, ValueError):
        LOGGER.debug("Evaluation telemetry delivery failed; continuing without it.")
        return False
    except Exception:
        LOGGER.debug("Evaluation telemetry failed unexpectedly; continuing without it.")
        return False
