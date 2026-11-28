"""Optional, background RAGAS scoring for EvidenceFlow runtime outcomes.

This module is disabled unless EVALUATION_RAGAS_ENABLED=true. When enabled, it
uses the short-lived Gateway session belonging to the current EvidenceFlow
request. Only normalized scalar scores and version labels are sent to the
evaluation harness; prompts, answers and retrieved contexts are never sent to
the harness. They are passed only to the configured Gateway for judging.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from urllib.parse import urlsplit
from typing import Any

from shared.evaluation_harness import record_event

LOGGER = logging.getLogger(__name__)
_EVALUATION_TASKS: set[asyncio.Task] = set()


def ragas_evaluation_enabled() -> bool:
    return os.getenv("EVALUATION_RAGAS_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _trusted_gateway_url(value: str) -> str | None:
    candidate = str(value or "").strip().rstrip("/")
    if not candidate or len(candidate) > 500:
        return None
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        local_http = parsed.scheme == "http" and hostname in {
            "localhost", "127.0.0.1", "::1"
        }
        if (
            not hostname
            or (parsed.scheme != "https" and not local_http)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.port not in ({None, 443} if parsed.scheme == "https" else {None, 80, 8000})
        ):
            return None
        return candidate
    except ValueError:
        return None


def _score_value(result: Any) -> float | None:
    raw = getattr(result, "value", result)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return round(value, 6)


async def evaluate_rag_with_ragas(
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    gateway_token: str,
    gateway_url: str,
    model_label: str | None = None,
    experiment_id: str | None = None,
    dataset_version: str | None = None,
    prompt_version: str | None = None,
    config_version: str | None = None,
    timeout_seconds: float = 45.0,
) -> dict[str, float]:
    """Run opt-in RAGAS Faithfulness and Context Precision against a Gateway.

    The first live integration intentionally avoids embedding-dependent metrics.
    A separately configured embedding model can be added without changing the
    content-free event contract.
    """
    trusted_url = _trusted_gateway_url(gateway_url)
    token = str(gateway_token or "").strip()
    query = str(user_input or "").strip()[:12_000]
    answer = str(response or "").strip()[:4_000]
    contexts = [
        str(item).strip()[:3_000]
        for item in (retrieved_contexts or [])[:8]
        if isinstance(item, str) and item.strip()
    ]
    if not trusted_url or not token or not query or not answer or not contexts:
        return {}

    client = None
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness, ContextPrecisionWithoutReference

        client = AsyncOpenAI(
            api_key=token,
            base_url=trusted_url,
            timeout=min(max(float(timeout_seconds), 5.0), 60.0),
            max_retries=0,
        )
        evaluator_llm = llm_factory("session-model", client=client)
        faithfulness = Faithfulness(llm=evaluator_llm)
        context_precision = ContextPrecisionWithoutReference(llm=evaluator_llm)

        faithfulness_result, precision_result = await asyncio.wait_for(
            asyncio.gather(
                faithfulness.ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                context_precision.ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
            ),
            timeout=min(max(float(timeout_seconds), 5.0), 60.0),
        )
        metrics: dict[str, float] = {}
        faithfulness_score = _score_value(faithfulness_result)
        precision_score = _score_value(precision_result)
        if faithfulness_score is not None:
            metrics["external.ragas.faithfulness"] = faithfulness_score
        if precision_score is not None:
            metrics["external.ragas.context_precision"] = precision_score
        if not metrics:
            return {}

        await asyncio.to_thread(
            record_event,
            event_type="evaluation.completed",
            status="success",
            workflow="rag_query",
            model=model_label,
            metrics=metrics,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            prompt_version=prompt_version,
            model_version=model_label,
            config_version=config_version,
        )
        return metrics
    except Exception as exc:  # noqa: BLE001 - judging must never alter user response.
        LOGGER.info("optional_ragas_evaluation_skipped error_type=%s", type(exc).__name__)
        return {}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                LOGGER.debug("optional_ragas_client_close_failed")


def _track_task(task: asyncio.Task) -> None:
    _EVALUATION_TASKS.add(task)
    task.add_done_callback(_EVALUATION_TASKS.discard)


def schedule_ragas_evaluation(
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    gateway_token: str,
    gateway_url: str,
    model_label: str | None = None,
    experiment_id: str | None = None,
    dataset_version: str | None = None,
    prompt_version: str | None = None,
    config_version: str | None = None,
) -> bool:
    """Schedule evaluator work after the user answer is ready; never block it."""
    if not ragas_evaluation_enabled():
        return False
    if not _trusted_gateway_url(gateway_url) or not gateway_token:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    async def run_safely() -> None:
        await evaluate_rag_with_ragas(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            gateway_token=gateway_token,
            gateway_url=gateway_url,
            model_label=model_label,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            prompt_version=prompt_version,
            config_version=config_version,
        )

    try:
        _track_task(loop.create_task(run_safely(), name="evidenceflow-ragas-evaluation"))
        return True
    except RuntimeError:
        return False
