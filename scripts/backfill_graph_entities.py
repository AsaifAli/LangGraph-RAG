"""One-time backfill for entity edges on documents indexed before GraphRAG.

Run with --dry-run first (the default), then --apply to update the configured
tenant index. This script stores only extracted entity terms; passage text is
never copied into update payloads or printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from retrieval.graph_rag import extract_entity_terms
from retrieval import rag_pipeline


def build_backfill_query(
    *,
    owner_scope: str | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"exists": {"field": "passage_text"}}]
    if owner_scope:
        filters.append({
            "term": {"owner_scope": rag_pipeline._validated_owner_scope(owner_scope)}
        })
    authorised_ids = list(dict.fromkeys(
        str(document_id).strip()
        for document_id in (document_ids or [])
        if str(document_id).strip()
    ))
    if authorised_ids:
        filters.append({"terms": {"document_id": authorised_ids}})
    return {"query": {"bool": {"filter": filters}}}


def backfill_entity_terms(
    client: Any,
    index_name: str,
    *,
    batch_size: int = 250,
    dry_run: bool = True,
    owner_scope: str | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, int]:
    """Backfill missing entity_terms with bounded bulk updates."""
    if not index_name or not str(index_name).strip():
        raise ValueError("A valid OpenSearch index name is required.")
    if not 1 <= int(batch_size) <= 5000:
        raise ValueError("batch_size must be between 1 and 5000.")

    from opensearchpy.helpers import bulk, scan

    stats = {
        "scanned": 0,
        "eligible": 0,
        "updated": 0,
        "skipped_existing": 0,
        "without_entities": 0,
    }
    query = build_backfill_query(
        owner_scope=owner_scope,
        document_ids=document_ids,
    )
    actions: list[dict[str, Any]] = []

    def flush() -> None:
        if not actions or dry_run:
            actions.clear()
            return
        successful, _errors = bulk(
            client,
            actions,
            chunk_size=batch_size,
            raise_on_error=True,
        )
        stats["updated"] += int(successful)
        actions.clear()

    for hit in scan(
        client,
        index=index_name,
        query=query,
        size=batch_size,
        scroll="5m",
        _source=["passage_text", "entity_terms"],
    ):
        stats["scanned"] += 1
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict):
            continue
        if "entity_terms" in source:
            stats["skipped_existing"] += 1
            continue
        text = source.get("passage_text")
        entities = extract_entity_terms(text if isinstance(text, str) else "")
        if not entities:
            stats["without_entities"] += 1
            continue

        stats["eligible"] += 1
        if dry_run:
            continue
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": str(hit.get("_id") or ""),
            "doc": {"entity_terms": entities},
        })
        if len(actions) >= batch_size:
            flush()

    flush()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-schema", default=rag_pipeline.TENANT_SCHEMA)
    parser.add_argument("--platform-tenant-id", default=rag_pipeline.PLATFORM_TENANT_ID)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--owner-scope", help="Optional verified owner scope to limit the backfill.")
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Optional document ID filter; may be repeated.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the backfill. Without this flag the command only reports a dry-run count.",
    )
    args = parser.parse_args(argv)

    client = rag_pipeline.build_opensearch_client()
    index_name = rag_pipeline.opensearch_index_name(
        args.tenant_schema,
        args.platform_tenant_id,
    )
    if not client.indices.exists(index=index_name):
        raise RuntimeError(
            f"OpenSearch index does not exist: {index_name}. "
            "Ingest at least one document before running the graph backfill."
        )
    # This migration needs only one keyword field; do not trigger optional
    # neural-sparse model registration/deployment as a side effect.
    rag_pipeline._ensure_entity_mapping(client, index_name)
    result = backfill_entity_terms(
        client,
        index_name,
        batch_size=args.batch_size,
        dry_run=not args.apply,
        owner_scope=args.owner_scope,
        document_ids=args.document_ids,
    )
    print(json.dumps({
        "index": index_name,
        "dry_run": not args.apply,
        **result,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

