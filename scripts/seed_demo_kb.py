"""Seed the synthetic portfolio demo documents into local Qdrant.

The demo content lives under ``demo/`` so the application code stays generic.
The two fixed document IDs make the benchmark reproducible. Re-running the
script deletes those two IDs first and then ingests the current demo files.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.rag_pipeline import (  # noqa: E402
    PLATFORM_TENANT_ID,
    TENANT_SCHEMA,
    delete_document,
    ingest_document,
)

DEMO_DOCUMENTS = [
    (_ROOT / "demo" / "policy_2025.md", "20252025-0000-0000-0000-000000000001"),
    (_ROOT / "demo" / "policy_2026.md", "20262026-0000-0000-0000-000000000002"),
]


async def seed() -> None:
    for path, document_id in DEMO_DOCUMENTS:
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            await delete_document(
                document_id,
                tenant_schema=TENANT_SCHEMA,
                platform_tenant_id=PLATFORM_TENANT_ID,
            )
        except Exception:
            # A first run may not have a collection yet; ingestion will create it.
            pass
        result = await ingest_document(
            path.read_bytes(),
            source_name=path.name,
            tenant_schema=TENANT_SCHEMA,
            platform_tenant_id=PLATFORM_TENANT_ID,
            document_id=document_id,
        )
        print(f"Seeded {path.name}: {result['document_id']} ({result['chunk_count']} chunks)")


if __name__ == "__main__":
    asyncio.run(seed())
