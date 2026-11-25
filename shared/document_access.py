"""Fail-closed ownership checks for EvidenceFlow's persisted document registry."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_OWNER_SCOPE_RE = re.compile(r"^[a-f0-9]{64}$")


def owner_scope_matches(record: dict[str, Any], owner_scope: str) -> bool:
    scope = str(owner_scope or "").strip().lower()
    if not _OWNER_SCOPE_RE.fullmatch(scope):
        return False
    return record.get("owner_scope") == scope


def filter_documents_for_owner(
    records: Iterable[dict[str, Any]],
    owner_scope: str,
) -> list[dict[str, Any]]:
    """Return only records whose server-assigned owner hash matches exactly.

    Legacy records without an owner are deliberately invisible; they are not
    claimed by the first authenticated visitor.
    """
    return [
        record
        for record in records
        if isinstance(record, dict) and owner_scope_matches(record, owner_scope)
    ]
