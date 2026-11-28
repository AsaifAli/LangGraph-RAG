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
    experiment_id: str | None = None,
    dataset_version: str | None = None,
    prompt_version: str | None = None,
    model_version: str | None = None,
    config_version: str | None = None,
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
    for key, value in (
        ("experiment_id", experiment_id),
        ("dataset_version", dataset_version),
        ("prompt_version", prompt_version),
        ("model_version", model_version),
        ("config_version", config_version),
    ):
        if isinstance(value, str) and value.strip():
            payload[key] = value.strip()[:128]

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



def _finite_metric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def build_runtime_evaluation_metrics(outcome: dict) -> dict[str, float | int]:
    """Project deterministic EvidenceFlow guardrail/evidence checks into scalar metrics.

    These are runtime checks, not LLM-as-a-judge scores. Only bounded numeric
    values leave the application; document content, prompts, final answers,
    source names, citations and guardrail text never enter the event payload.
    """
    if not isinstance(outcome, dict):
        return {}

    metrics: dict[str, float | int] = {}

    quality = outcome.get("quality") if isinstance(outcome.get("quality"), dict) else {}
    for source, target in (
        ("citation_coverage", "runtime.citation_coverage"),
        ("citation_verification", "runtime.citation_verification"),
    ):
        value = quality.get(source)
        if _finite_metric(value) and 0.0 <= float(value) <= 100.0:
            metrics[target] = round(float(value) / 100.0, 6)

    grounding_status = str(outcome.get("grounding_status") or "").upper()
    if grounding_status in {"VERIFIED", "PARTIAL", "UNVERIFIED", "NOT_APPLICABLE"}:
        metrics["runtime.grounding_verified"] = float(grounding_status == "VERIFIED")
        metrics["runtime.grounding_partial"] = float(grounding_status == "PARTIAL")

    trust = outcome.get("trust") if isinstance(outcome.get("trust"), dict) else {}
    trust_label = str(trust.get("label") or "").upper()
    if trust_label:
        metrics["runtime.trust_verified"] = float(trust_label == "VERIFIED")
        metrics["runtime.trust_review_required"] = float(trust_label == "REVIEW")
        metrics["runtime.security_blocked"] = float(
            trust.get("security_blocked") is True or trust_label == "BLOCKED"
        )

    if isinstance(outcome.get("abstained"), bool):
        metrics["runtime.abstained"] = float(outcome["abstained"])

    numeric_support = quality.get("numeric_support")
    if isinstance(numeric_support, dict) and isinstance(
        numeric_support.get("numeric_claims_supported"), bool
    ):
        metrics["runtime.numeric_claims_supported"] = float(
            numeric_support["numeric_claims_supported"]
        )

    for source, target in (
        ("verified_count", "evidence.verified_count"),
        ("proposed_count", "evidence.proposed_count"),
    ):
        value = outcome.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[target] = min(value, 2_147_483_647)

    chunks = outcome.get("chunks") if isinstance(outcome.get("chunks"), list) else []
    metrics["retrieval.chunk_count"] = min(len(chunks), 2_147_483_647)
    graph_expanded = 0
    for chunk in chunks:
        if isinstance(chunk, dict):
            metadata = chunk.get("meta_data")
        else:
            metadata = getattr(chunk, "meta_data", None)
        if isinstance(metadata, dict) and metadata.get("graph_expanded") is True:
            graph_expanded += 1
    metrics["retrieval.graph_expanded_chunk_count"] = graph_expanded

    web_sources = outcome.get("web_sources")
    if isinstance(web_sources, list):
        metrics["retrieval.web_source_count"] = min(len(web_sources), 2_147_483_647)

    findings = outcome.get("guardrail_findings")
    if isinstance(findings, list):
        metrics["runtime.guardrail_finding_count"] = min(len(findings), 2_147_483_647)
        metrics["runtime.guardrail_blocked"] = float(
            any(isinstance(item, dict) and item.get("blocked") is True for item in findings)
        )

    return metrics
