"""Builds the Qdrant filter every knowledge-base retrieval call uses:
tenant + document scoping (always applied) plus optional in-chat thread
scoping and person/org entity narrowing. `scope` is typed `Any` rather than
a dedicated class since this app has no in-chat thread-scoping caller yet —
adding one later just means passing an object with `.document_ids`/`.mode`/
`.session_thread_filter` attributes, no signature change needed.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from shared.constants import VectorDBConstants
from retrieval.entity_filters import entity_coexistence_conditions
from shared.entities import SecureAgentRequest


def build_kb_filter_expr(
    request: SecureAgentRequest,
    *,
    scope: Any | None = None,
) -> models.Filter:
    """Build tenant + document filter with optional in-chat thread scoping."""
    if scope is not None and scope.document_ids is not None:
        effective_document_ids = scope.document_ids
    else:
        effective_document_ids = request.document_ids or []

    must: list = [
        models.FieldCondition(
            key=VectorDBConstants.META_TENANT_SCHEMA_KEY,
            match=models.MatchValue(
                value=f"{request.tenant_schema}_{request.platform_tenant_id}"
            ),
        ),
        models.FieldCondition(
            key=VectorDBConstants.META_DOCUMENT_ID_KEY,
            match=models.MatchAny(
                any=[str(doc_id) for doc_id in effective_document_ids]
            ),
        ),
    ]

    if (
        scope is not None
        and scope.mode == "chat"
        and scope.session_thread_filter
    ):
        must.append(
            models.FieldCondition(
                key=VectorDBConstants.META_THREAD_ID_KEY,
                match=models.MatchValue(value=scope.session_thread_filter),
            )
        )

    must.extend(
        entity_coexistence_conditions(request.person_ids, request.org_ids)
    )

    return models.Filter(must=must)
