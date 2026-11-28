"""Deterministic source-selection guardrails for EvidenceFlow routing.

Uploaded documents are evidence about their own contents and user-provided
records. They are not a substitute for public, current or independently
verifiable facts. This classifier handles explicit source intent; ambiguous
requests remain with the LangGraph model router.
"""
from __future__ import annotations

import re


_EXPLICIT_WEB = re.compile(
    r"\b(?:search(?:\s+the)?\s+web|web\s+search|search\s+online|look\s+online|"
    r"on\s+the\s+internet|from\s+the\s+internet|websites?|online\s+sources?)\b",
    re.IGNORECASE,
)
_DOCUMENT_CONTEXT = re.compile(
    r"\b(?:uploaded|attached|this\s+(?:(?:supplier|vendor|service)\s+)?"
    r"(?:document|file|quotation|quote|proposal|contract|spreadsheet)|"
    r"my\s+(?:document|file|quotation|quote|proposal)|our\s+(?:document|quotation|quote|proposal)|"
    r"in\s+the\s+(?:document|file|quotation|quote)|"
    r"the\s+provided\s+(?:document|file|quotation|quote)|"
    r"(?:supplier|vendor)\s+(?:quotation|quote|proposal))\b",
    re.IGNORECASE,
)
_EXTERNAL_FACTS = re.compile(
    r"\b(?:current(?:ly)?|latest|today|yesterday|right\s+now|as\s+of\s+now|"
    r"recent(?:ly)?|this\s+(?:week|month|year)|market\s+(?:price|rate|average|benchmark)|"
    r"industry\s+(?:standard|benchmark|average)|benchmark(?:s|ing)?|"
    r"(?:going|typical|prevailing)\s+(?:market\s+)?rate|price\s+range|"
    r"competitive(?:ly)?\s+(?:price|priced)?|fair(?:ly)?\s+(?:price|priced|quote|quotation)|"
    r"reasonable\s+(?:market\s+)?(?:price|rate|quote|quotation)|"
    r"supplier\s+(?:reputation|reviews|credentials)|independently\s+verif(?:y|ied)|"
    r"public\s+records?|regulatory\s+(?:update|changes?)|latest\s+regulations?)\b",
    re.IGNORECASE,
)


def classify_source_intent(
    query: str,
    *,
    primary_source_available: bool = False,
) -> str:
    """Return ``documents``, ``web``, ``both`` or ``auto`` for a user query.

    Explicit document + external benchmark/verification requests use both
    sources. Explicit online requests use web. Explicit uploaded-document
    references use the KB. Current facts can be handed to the model router
    when the approved primary-source connector is available; otherwise they
    are sent to web research. Ambiguous/general requests remain model-routed.
    """
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    if not text:
        return "auto"

    explicit_web = bool(_EXPLICIT_WEB.search(text))
    document_context = bool(_DOCUMENT_CONTEXT.search(text))
    external_fact = bool(_EXTERNAL_FACTS.search(text))

    if document_context and (explicit_web or external_fact):
        return "both"
    if explicit_web:
        return "web"
    if document_context:
        return "documents"
    if external_fact and not primary_source_available:
        return "web"
    return "auto"
