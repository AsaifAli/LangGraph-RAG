"""Lightweight entity-co-occurrence graph primitives for sparse-first retrieval.

The graph is deliberately stored as keyword edges on existing OpenSearch
passages: a passage is connected to the entity terms it mentions, and passages
sharing an entity are adjacent. This is bounded GraphRAG-style expansion, not
an ontology or subject-predicate-object knowledge graph.
"""

from __future__ import annotations

import re
from collections import OrderedDict

_ENTITY_PHRASE = re.compile(
    r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){1,3}\b"
)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*\b")
_STANDARD = re.compile(r"\b([A-Z]{2,})\s+(\d{3,}(?:-\d+)?)\b")
_CAMEL_CASE = re.compile(
    r"\b(?=[A-Za-z0-9&'.-]*[a-z])(?=[A-Za-z0-9&'.-]*[A-Z])"
    r"[A-Za-z][A-Za-z0-9&'.-]+\b"
)
_TITLE_CASE = re.compile(r"\b[A-Z][a-z][A-Za-z0-9&'.-]{2,}\b")

_STOP_ENTITIES = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "before", "both", "but", "by", "can", "chapter", "clause",
    "company", "could", "document", "each", "either", "else", "figure",
    "for", "from", "furthermore", "has", "have", "he", "her", "here",
    "hers", "him", "his", "however", "if", "in", "into", "is", "it",
    "its", "may", "more", "must", "neither", "no", "not", "of", "on",
    "one", "or", "our", "page", "policy", "section", "she", "should",
    "since", "such", "table", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "under", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "would", "you", "your",
    "agreement", "agreements", "contract", "contracts", "schedule", "schedules",
    "exhibit", "appendix", "attachment", "attachments", "supplier", "suppliers",
    "customer", "customers", "services", "service", "payment", "payments",
    "whereas", "insured", "insurer", "provider", "providers", "parties",
    "party", "beneficiary", "beneficiaries", "liability", "coverage",
    "coverages", "warranty", "warranties", "declaration", "declarations",
    "conditions", "condition", "effective", "effective", "term", "terms",
    "amount", "amounts", "date", "dates", "notice", "notices", "premises",
    "policyholder", "policyholders", "usd", "inr", "eur", "gbp", "aud",
    "cad", "jpy", "chf", "cny",
}
_GENERIC_NAME_SUFFIXES = {
    "company", "corporation", "corp", "inc", "incorporated", "limited",
    "ltd", "llc", "plc", "group", "holdings",
}
_LEGAL_FORM_SUFFIXES = {
    "company", "corporation", "corp", "inc", "incorporated",
    "limited", "ltd", "llc", "plc",
}
_CURRENCY_CODES = {"usd", "inr", "eur", "gbp", "aud", "cad", "jpy", "chf", "cny"}


def extract_entity_terms(text: str, *, max_entities: int = 24) -> list[str]:
    """Extract bounded, normalized entity candidates without a model call.

    Multi-word proper-name candidates, acronyms, standards, and CamelCase
    names are retained. Common sentence starters and generic document labels
    are filtered to reduce noisy graph edges. This is intentionally a
    recall-oriented heuristic; the retrieval reranker remains the precision
    gate.
    """
    if max_entities <= 0 or not isinstance(text, str) or not text.strip():
        return []

    found: OrderedDict[str, None] = OrderedDict()

    def add(value: str) -> None:
        normalized = re.sub(r"\s+", " ", value).strip(" .,:;()[]{}'\"").casefold()
        if (
            normalized
            and len(normalized) <= 64
            and len(normalized) >= 2
            and normalized not in _STOP_ENTITIES
            and normalized not in _GENERIC_NAME_SUFFIXES
            and not normalized.isnumeric()
        ):
            found.setdefault(normalized, None)

    for match in _ENTITY_PHRASE.finditer(text):
        words = [
            word.strip(" .,:;()[]{}'\"")
            for word in match.group(0).split()
            if word.strip(" .,:;()[]{}'\"").casefold() not in _STOP_ENTITIES
        ]
        if len(words) >= 2:
            add(" ".join(words))
            canonical_words = list(words)
            while canonical_words and canonical_words[-1].casefold() in _LEGAL_FORM_SUFFIXES:
                canonical_words.pop()
            if len(canonical_words) >= 2:
                add(" ".join(canonical_words))
            for word in words:
                if len(word) >= 4 and word.casefold() not in _GENERIC_NAME_SUFFIXES:
                    add(word)

    for acronym, number in _STANDARD.findall(text):
        if acronym.casefold() not in _CURRENCY_CODES:
            add(acronym)
            add(f"{acronym} {number}")

    for match in _ACRONYM.finditer(text):
        add(match.group(0))

    for pattern in (_CAMEL_CASE, _TITLE_CASE):
        for match in pattern.finditer(text):
            add(match.group(0))

    return list(found.keys())[:max_entities]

