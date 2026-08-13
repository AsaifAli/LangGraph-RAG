"""Deterministic answer-quality and evidence-safety helpers.

The goal is not to pretend that deterministic checks prove semantic truth. They
measure what the application can prove without another LLM call: citation
coverage, numeric/date support, evidence conflicts, and whether a KB answer
has enough authorized evidence to be shown as grounded.
"""
from __future__ import annotations

import re
from typing import Iterable

from citations.verifier import values_supported

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or",
    "in", "on", "for", "with", "per", "from", "by", "this", "that", "it", "as",
    "what", "which", "how", "does", "do", "has", "have", "had", "than", "then",
}


def citation_quality(*, proposed_count: int, verified_count: int, grounding_status: str | None) -> dict:
    """Return deterministic citation QA metrics on a 0-100 scale."""
    if proposed_count <= 0:
        return {
            "citation_coverage": None,
            "citation_verification": None,
            "quality_label": "NO_KB_EVIDENCE",
        }

    coverage = round(min(verified_count / proposed_count, 1.0) * 100, 1)
    verification = coverage if grounding_status in {"VERIFIED", "PARTIAL", "UNVERIFIED"} else None
    if grounding_status == "VERIFIED":
        label = "HIGH"
    elif grounding_status == "PARTIAL":
        label = "REVIEW"
    else:
        label = "LOW"
    return {
        "citation_coverage": coverage,
        "citation_verification": verification,
        "quality_label": label,
    }


def numeric_support(answer: str, evidence_texts: Iterable[str]) -> dict:
    """Check material numbers/dates in an answer against cited evidence."""
    ok, missing = values_supported(answer or "", list(evidence_texts or []))
    return {
        "numeric_claims_supported": ok,
        "unsupported_values": sorted(missing),
    }


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUM_RE.findall(text or "")}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z_-]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def evidence_conflict_candidates(answer: str, evidence_refs: Iterable[dict]) -> list[dict]:
    """Find *possible* numeric conflicts across cited evidence.

    This is deliberately labelled a candidate detector, not a contradiction
    proof. It flags an answer sentence containing a number when two cited
    evidence passages share meaningful keywords with that sentence but expose
    different numeric values. A later semantic evaluator can confirm whether
    the values really describe the same field.
    """
    refs = list(evidence_refs or [])
    candidates: list[dict] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer or ""):
        answer_nums = _numbers(sentence)
        if not answer_nums:
            continue
        keys = _keywords(sentence)
        if not keys:
            continue
        matches = []
        for ref in refs:
            content = str(ref.get("content") or "")
            overlap = keys & _keywords(content)
            if len(overlap) >= 2:
                nums = _numbers(content)
                if nums:
                    matches.append((str(ref.get("evidence_id") or ""), nums, overlap))
        for i, left in enumerate(matches):
            for right in matches[i + 1:]:
                if left[1] != right[1] and left[0] and right[0]:
                    candidates.append({
                        "sentence": sentence.strip(),
                        "evidence_ids": [left[0], right[0]],
                        "values": [sorted(left[1]), sorted(right[1])],
                        "shared_terms": sorted(left[2] & right[2]),
                    })
    # De-duplicate equivalent candidates for a compact UI.
    unique = {}
    for item in candidates:
        key = (item["sentence"], tuple(item["evidence_ids"]))
        unique[key] = item
    return list(unique.values())


def should_abstain_from_kb(*, kb_requested: bool, chunk_count: int, verified_count: int, web_used: bool) -> bool:
    """Fail closed when a KB route produced no usable evidence.

    Web-only turns are excluded because their evidence model is separate.
    """
    return bool(kb_requested and not web_used and (chunk_count == 0 or verified_count == 0))
