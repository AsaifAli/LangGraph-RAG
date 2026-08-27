"""OpenSearch scope-filter helper.

Kept as a small compatibility module for callers that previously imported the
provider-specific filter builder. The new retrieval implementation builds these
filters directly as OpenSearch Query DSL clauses.
"""

from __future__ import annotations

from typing import Any


def build_kb_filter_expr(request: Any, scope: Any = None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"term": {"tenant_schema": f"{request.tenant_schema}_{request.platform_tenant_id}"}},
        {"term": {"platform_tenant_id": request.platform_tenant_id}},
    ]
    document_ids = getattr(request, "document_ids", None)
    if document_ids:
        filters.append({"terms": {"document_id": [str(x) for x in document_ids]}})
    return filters
