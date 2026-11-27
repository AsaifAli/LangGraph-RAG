"""Recognize references that look like a named document, independent of parsing."""
from __future__ import annotations

import re

FILENAME_LIKE_RE = re.compile(
    r"\b[\w\-. ]+\.(?:txt|pdf|docx?|xlsx?|xlsm|xlsb|ods|pptx?|odp|odt|rtf|csv|tsv|md|json|xml|ya?ml|html?)\b",
    re.IGNORECASE,
)


def contains_document_filename(query: str) -> bool:
    return bool(FILENAME_LIKE_RE.search(query or ""))
