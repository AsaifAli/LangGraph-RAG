"""Shared Qdrant filter conditions for canonical-entity retrieval: narrows
results to chunks tagged with a given person/org id, while never excluding
older chunks that predate canonical-id tagging entirely (those carry no
person/org payload at all, so they'd otherwise be silently dropped by an
id filter that only matched on presence).
"""
from typing import Iterable, Optional

from qdrant_client.http import models

from shared.constants import VectorDBConstants


def _coexistence_clause(key: str, ids: list[str]) -> models.Filter:
    """A chunk passes if it has no payload for *key* (legacy, untagged) OR its
    payload matches one of *ids*."""
    return models.Filter(
        should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
            models.FieldCondition(key=key, match=models.MatchAny(any=ids)),
        ]
    )


def entity_coexistence_conditions(
    person_ids: Optional[Iterable[str]],
    org_ids: Optional[Iterable[str]],
) -> list[models.Filter]:
    """Return nested ``must`` filter clauses narrowing retrieval to canonical
    entity id(s), while never excluding legacy documents that predate the
    canonical-id design (those carry no person/org id payload).
    """
    conditions: list[models.Filter] = []

    pids = [str(p) for p in (person_ids or [])]
    if pids:
        conditions.append(
            _coexistence_clause(VectorDBConstants.META_PERSON_IDS_KEY, pids)
        )

    oids = [str(o) for o in (org_ids or [])]
    if oids:
        conditions.append(
            _coexistence_clause(VectorDBConstants.META_ORG_IDS_KEY, oids)
        )

    return conditions
