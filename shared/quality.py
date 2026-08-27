"""Deterministic answer-quality and evidence-safety helpers.

The goal is not to pretend that deterministic checks prove semantic truth. They
measure what the application can prove without another LLM call: citation
coverage, numeric/date support, evidence conflicts, and whether a KB answer
has enough authorized evidence to be shown as grounded.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
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


def _number_mentions(text: str) -> list[tuple[str, set[str]]]:
    """Return normalized numeric tokens with local semantic context.

    Context is deliberately small and keyword-based. The detector is a safety
    heuristic, not a semantic truth oracle, so it should only compare numbers
    when the surrounding claim vocabulary is strongly aligned.
    """
    text = text or ""
    matches = list(_NUM_RE.finditer(text))
    mentions: list[tuple[str, set[str]]] = []
    for match in matches:
        value = match.group(0).replace(",", "")
        left = max(0, match.start() - 120)
        right = min(len(text), match.end() + 120)
        context = _keywords(text[left:right])
        mentions.append((value, context))
    return mentions


def _context_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def _evidence_overlap_groups(evidence_refs: Iterable[dict]) -> list[dict]:
    """Identify near-duplicate evidence passages without treating them as conflicts."""
    refs = list(evidence_refs or [])
    groups: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, left in enumerate(refs):
        left_id = str(left.get("evidence_id") or "")
        left_doc = str(left.get("document_id") or "")
        left_keys = _keywords(str(left.get("content") or ""))
        if not left_id or not left_keys:
            continue
        for right in refs[i + 1:]:
            right_id = str(right.get("evidence_id") or "")
            pair = tuple(sorted((left_id, right_id)))
            if not right_id or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            right_content = str(right.get("content") or "")
            normalized_left = re.sub(r"\s+", " ", str(left.get("content") or "").strip().lower())
            normalized_right = re.sub(r"\s+", " ", right_content.strip().lower())
            same_doc = bool(left_doc and left_doc == str(right.get("document_id") or ""))
            text_similarity = SequenceMatcher(None, normalized_left, normalized_right).ratio()
            left_numbers = _numbers(normalized_left)
            right_numbers = _numbers(normalized_right)
            # Two passages that differ only in a number are NOT benign overlap:
            # they are precisely the pattern a contradiction detector must inspect.
            # Only classify near-duplicates as overlap when their numeric sets also
            # agree.
            threshold = 0.90 if same_doc else 0.94
            if (normalized_left == normalized_right or text_similarity >= threshold) and left_numbers == right_numbers:
                groups.append({
                    "evidence_ids": [left_id, right_id],
                    "document_ids": [left_doc, str(right.get("document_id") or "")],
                    "similarity": round(text_similarity, 3),
                })
    return groups


def evidence_consistency(answer: str, evidence_refs: Iterable[dict]) -> dict[str, list[dict]]:
    """Separate genuine numeric conflicts from benign evidence overlap.

    A conflict requires two cited passages to discuss the *same numeric claim*
    with materially different values. Generic keyword overlap is not enough.
    Near-duplicate passages are reported separately as overlap and never counted
    as contradictions.
    """
    refs = list(evidence_refs or [])
    overlaps = _evidence_overlap_groups(refs)
    overlap_pairs = {
        tuple(sorted(item["evidence_ids"]))
        for item in overlaps
        if len(item.get("evidence_ids", [])) == 2
    }

    conflicts: list[dict] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer or ""):
        answer_mentions = _number_mentions(sentence)
        if not answer_mentions:
            continue

        evidence_mentions: dict[str, list[tuple[str, set[str]]]] = {}
        for ref in refs:
            eid = str(ref.get("evidence_id") or "")
            content = str(ref.get("content") or "")
            if eid and content:
                evidence_mentions[eid] = _number_mentions(content)

        for answer_value, answer_context in answer_mentions:
            matches: list[tuple[str, str, float]] = []
            for eid, mentions in evidence_mentions.items():
                best: tuple[str, float] | None = None
                for evidence_value, evidence_context in mentions:
                    similarity = _context_similarity(answer_context, evidence_context)
                    if similarity < 0.50:
                        continue
                    candidate = (evidence_value, similarity)
                    if best is None or similarity > best[1]:
                        best = candidate
                if best is not None:
                    matches.append((eid, best[0], best[1]))

            # A true contradiction exists only when the SAME answer-number
            # context is supported by at least two cited passages with different
            # values, and at least one of those values matches the answer. This
            # is the key guard against false positives such as an answer citing
            # "1-2+ years" while another paragraph in the same JD contains "4"
            # agents, a page number, or a different business metric.
            answer_support = [m for m in matches if m[1] == answer_value]
            alternate_support = [m for m in matches if m[1] != answer_value]
            for answer_id, _, answer_sim in answer_support:
                for alternate_id, alternate_value, alternate_sim in alternate_support:
                    pair = tuple(sorted((answer_id, alternate_id)))
                    if pair in overlap_pairs:
                        continue
                    conflicts.append({
                        "sentence": sentence.strip(),
                        "evidence_ids": [answer_id, alternate_id],
                        "answer_value": answer_value,
                        "values": sorted({answer_value, alternate_value}),
                        "shared_terms": sorted(answer_context),
                        "context_similarity": round(min(answer_sim, alternate_sim), 3),
                    })

    unique: dict[tuple, dict] = {}
    for item in conflicts:
        key = (item["sentence"], tuple(sorted(item["evidence_ids"])), tuple(item["values"]))
        unique[key] = item
    return {"conflicts": list(unique.values()), "overlaps": overlaps}


def evidence_conflict_candidates(answer: str, evidence_refs: Iterable[dict]) -> list[dict]:
    """Backward-compatible contradiction-only view of :func:`evidence_consistency`."""
    return evidence_consistency(answer, evidence_refs)["conflicts"]


def should_abstain_from_kb(*, kb_requested: bool, chunk_count: int, verified_count: int, web_used: bool) -> bool:
    """Fail closed when a KB route produced no usable evidence.

    Web-only turns are excluded because their evidence model is separate.
    """
    return bool(kb_requested and not web_used and (chunk_count == 0 or verified_count == 0))
