"""Ingest a local document into the POC's demo Qdrant collection, so
custom_langgraph_poc.py can retrieve from it.

CLI counterpart to streamlit_app.py's upload widgets — both call the same
`rag_pipeline.ingest_document`, so behavior is identical either way.
Text extraction (PDF/DOCX/DOC) is handled by `shared/text_extractors.py`;
chunking is a simple recursive-splitter chunker — see `chunk_text`'s
docstring in rag_pipeline.py for the tradeoffs.

Run (from poc/langgraph_rag/):
    python scripts/ingest_document_poc.py path/to/document.txt
    python scripts/ingest_document_poc.py path/to/document.pdf --document-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# This file lives in scripts/ — POC root is one level up, needed so
# `python scripts/ingest_document_poc.py` resolves `from retrieval...` below
# regardless of CWD.
_POC_ROOT = Path(__file__).resolve().parent.parent
if str(_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_POC_ROOT))

from retrieval.rag_pipeline import PLATFORM_TENANT_ID, TENANT_SCHEMA, ingest_document  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    path = Path(args.file)
    content = path.read_bytes()
    result = await ingest_document(
        content,
        source_name=path.name,
        tenant_schema=args.tenant_schema,
        platform_tenant_id=args.platform_tenant_id,
        document_id=args.document_id,
    )
    print(f"Ingested {result['chunk_count']} chunks from {path.name}")
    print(f"document_id: {result['document_id']}")
    print(
        f"Use --document-ids {result['document_id']} with custom_langgraph_poc.py "
        "to query it (comma-separate with other document ids to query multiple "
        "documents at once)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to a local .txt/.md/.csv/.pdf/.docx/.doc file")
    parser.add_argument("--tenant-schema", default=TENANT_SCHEMA)
    parser.add_argument("--platform-tenant-id", default=PLATFORM_TENANT_ID)
    parser.add_argument(
        "--document-id",
        default=None,
        help="Fixed document id to use (default: a new random UUID each run)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
