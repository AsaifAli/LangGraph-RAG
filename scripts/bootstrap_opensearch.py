"""Bootstrap the local OpenSearch sparse-first retrieval stack.

Run from the project root after `docker compose up -d`:
    python scripts/bootstrap_opensearch.py

The operation is idempotent. It creates the OpenSearch index and, when
OPENSEARCH_NEURAL_SPARSE=true, registers/deploys the local OpenSearch neural
sparse models used for semantic sparse retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.rag_pipeline import (  # noqa: E402
    build_opensearch_client,
    ensure_opensearch_index,
    opensearch_index_name,
    PLATFORM_TENANT_ID,
    TENANT_SCHEMA,
)


def main() -> None:
    client = build_opensearch_client()
    index_name = opensearch_index_name(TENANT_SCHEMA, PLATFORM_TENANT_ID)
    ensure_opensearch_index(client, index_name=index_name)
    mapping = client.indices.get_mapping(index=index_name)
    properties = ((mapping.get(index_name) or {}).get("mappings") or {}).get("properties") or {}
    print(f"OpenSearch ready: {index_name}")
    print(f"Neural sparse field present: {'passage_embedding' in properties}")


if __name__ == "__main__":
    main()
