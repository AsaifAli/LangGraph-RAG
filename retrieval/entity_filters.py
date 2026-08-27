"""Entity-filter helpers kept provider-neutral for the new OpenSearch backend."""

from __future__ import annotations

from typing import Any


def entity_coexistence_conditions(person_ids: list[str] | None = None, org_ids: list[str] | None = None) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if person_ids:
        conditions.append({"terms": {"person_ids": person_ids}})
    if org_ids:
        conditions.append({"terms": {"org_ids": org_ids}})
    return conditions
