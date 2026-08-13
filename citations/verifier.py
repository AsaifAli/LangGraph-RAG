"""Stateless, deterministic citation verifier: checks that every citation an
agent emits actually traces back to a chunk it was really given, rather
than trusting the model's own claim of what it cited.

  * `verify_reference` validates ONE emitted evidence ref against the
    turn's CLOSED evidence registry: the evidence_id must be in the
    retrieved set, and the cited excerpt must match the EXACT evidence
    chunk's content (a substring/hash check against that specific chunk,
    not "any retrieved chunk"). Returns a machine-readable GroundingStatus
    + ReasonCode.
  * `finalize` assembles the FINAL citation set under a mode:
        - "hybrid"   -> passes proposed + heuristic refs through unchanged
                        (this module is a no-op advisor in this mode),
        - "verified" -> EVIDENCE-ONLY and FAIL-CLOSED: only VERIFIED refs
                        survive, heuristic extras are dropped,
                        unsupported/unverifiable refs are removed, and an
                        overall grounding_status is returned. Nothing
                        invented can be authorized.

Pure and deterministic — `python verifier.py` runs the self-tests. No agent state, no I/O.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


class GroundingStatus:
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ReasonCode:
    OK = "OK"
    ID_NOT_RETRIEVED = "ID_NOT_RETRIEVED"        # evidence_id absent from the retrieved closed set
    NO_EVIDENCE_ID = "NO_EVIDENCE_ID"            # ref carries no evidence_id to authorize
    DOCUMENT_MISMATCH = "DOCUMENT_MISMATCH"      # ref document_id != the evidence id's registry document
    EXCERPT_MISMATCH = "EXCERPT_MISMATCH"        # cited excerpt not contained in THAT evidence's chunk
    EXCERPT_UNCONFIRMED = "EXCERPT_UNCONFIRMED"  # excerpt not substring-confirmable (PII round-trip) -> PARTIAL, not VERIFIED
    EMPTY_REGISTRY = "EMPTY_REGISTRY"            # no closed evidence set for this turn


def _norm(text: str) -> str:
    """Formatting-only normalization for excerpt comparison (whitespace/case)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def excerpt_hash(text: str) -> str:
    """Stable hash of the normalized excerpt (presentation-integrity check)."""
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


# --- deterministic value/date validation (recommendation step 3, no LLM) -------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _material_numbers(text: str) -> set:
    """Numbers/currency amounts in *text*, comma-normalized. Numbers are cleartext across the PII
    round-trip (only names/emails/etc. are tokenized), so they compare reliably claim↔evidence."""
    return {m.replace(",", "") for m in _NUM_RE.findall(text or "") if any(c.isdigit() for c in m)}


def values_supported(claim_text: str, evidence_texts) -> tuple:
    """Every material NUMBER (and ISO/‘/’ date) in the claim must appear in its cited evidence content.
    Deterministic, no LLM. Returns (ok, missing_values). A claim with no numbers is trivially ok
    (this check is for figures/limits/dates; entailment of prose is a separate LLM step)."""
    claim_nums = _material_numbers(claim_text)
    claim_dates = set(_DATE_RE.findall(claim_text or ""))
    if not claim_nums and not claim_dates:
        return True, set()
    ev = " ".join(evidence_texts or [])
    ev_nums = _material_numbers(ev)
    ev_dates = set(_DATE_RE.findall(ev))
    missing = {n for n in claim_nums if n not in ev_nums} | {d for d in claim_dates if d not in ev_dates}
    return (not missing), missing


@dataclass
class VerifiedRef:
    document_id: str
    evidence_id: str
    content: str
    status: str
    reason: str
    retrieval_score: Optional[float] = None

    def as_dict(self) -> dict:
        return {"document_id": self.document_id, "evidence_id": self.evidence_id,
                "content": self.content, "verification_status": self.status,
                "verification_reason": self.reason, "retrieval_score": self.retrieval_score}


@dataclass
class FinalizeResult:
    references: List[dict] = field(default_factory=list)
    grounding_status: str = GroundingStatus.NOT_APPLICABLE
    reasons: Dict[str, int] = field(default_factory=dict)


# --- registry accessors (registry = {eid_to_doc, eid_to_content, eid_to_meta}) ------------------
def _reg_doc(registry: dict, eid: str) -> Optional[str]:
    return (registry.get("eid_to_doc") or {}).get(eid)


def _reg_content(registry: dict, eid: str) -> Optional[str]:
    return (registry.get("eid_to_content") or {}).get(eid)


def verify_reference(ref: dict, registry: dict, *, min_excerpt_chars: int = 24) -> VerifiedRef:
    """Validate one emitted evidence ref against the closed registry. `ref` = {document_id,
    evidence_id, content, retrieval_score?}. Fail-CLOSED: anything not positively verified is
    UNVERIFIED with a reason code."""
    eid = str(ref.get("evidence_id") or "").strip()
    did = str(ref.get("document_id") or "")
    content = ref.get("content") or ""
    score = ref.get("retrieval_score")
    if not eid:
        return VerifiedRef(did, eid, content, GroundingStatus.UNVERIFIED, ReasonCode.NO_EVIDENCE_ID, score)
    reg_did = _reg_doc(registry, eid)
    if not reg_did:
        return VerifiedRef(did, eid, content, GroundingStatus.UNVERIFIED, ReasonCode.ID_NOT_RETRIEVED, score)
    # IDENTITY: a valid evidence_id paired with the WRONG document is a mis-attribution, not support
    # (post-review F-02). The document must equal the id's registry document.
    if did and str(reg_did) != did:
        return VerifiedRef(did, eid, content, GroundingStatus.UNVERIFIED, ReasonCode.DOCUMENT_MISMATCH, score)
    # INTEGRITY (CIT-04): the cited excerpt should come from THIS evidence id's chunk. A substring hit
    # proves support -> VERIFIED. A miss is usually a PII-representation artifact (the ref content is
    # registry-sourced then restored/truncated, reg_content is raw surrogate), so the citation is KEPT
    # — but only as PARTIAL/EXCERPT_UNCONFIRMED, NEVER VERIFIED: identity is proven, SUPPORT is not
    # (post-review F-02: an unconfirmable excerpt must not be labeled verified).
    reg_content = _reg_content(registry, eid)
    ne = _norm(content)
    if reg_content is not None and len(ne) >= min_excerpt_chars and ne not in _norm(reg_content):
        return VerifiedRef(did, eid, content, GroundingStatus.PARTIAL, ReasonCode.EXCERPT_UNCONFIRMED, score)
    return VerifiedRef(did, eid, content, GroundingStatus.VERIFIED, ReasonCode.OK, score)


def finalize(evidence_refs: Sequence[dict], heuristic_refs: Sequence[dict], registry: dict,
             *, mode: str = "hybrid") -> FinalizeResult:
    """Assemble the final citation set.

    mode="verified": EVIDENCE-ONLY, FAIL-CLOSED. Verify every evidence ref; keep only VERIFIED ones;
      drop ALL heuristic extras. grounding_status = VERIFIED (all kept & none dropped),
      PARTIAL (some kept, some dropped), UNVERIFIED (none kept though refs were proposed),
      NOT_APPLICABLE (nothing proposed at all).
    mode="hybrid": advisory only — returns the union unchanged (caller keeps current behaviour).
    """
    reasons: Dict[str, int] = {}
    if mode != "verified":
        merged = list(evidence_refs) + [r for r in heuristic_refs]
        return FinalizeResult(merged, GroundingStatus.NOT_APPLICABLE, reasons)

    if not registry or not (registry.get("eid_to_doc")):
        reasons[ReasonCode.EMPTY_REGISTRY] = 1
        return FinalizeResult([], GroundingStatus.NOT_APPLICABLE, reasons)

    kept, dropped, partials = [], 0, 0
    proposed = list(evidence_refs or [])
    for ref in proposed:
        v = verify_reference(ref, registry)
        reasons[v.reason] = reasons.get(v.reason, 0) + 1
        if v.status == GroundingStatus.VERIFIED:
            kept.append(v.as_dict())
        elif v.status == GroundingStatus.PARTIAL:
            kept.append(v.as_dict())  # KEPT (cited) but not fully verified -> caps overall status
            partials += 1
        else:
            dropped += 1

    # Overall status is VERIFIED only when EVERY proposed ref fully verified (no partial/unconfirmed,
    # none dropped) — this is what a truthful "verified" mode label must require (post-review F-01/F-02).
    if not proposed:
        status = GroundingStatus.NOT_APPLICABLE
    elif kept and dropped == 0 and partials == 0:
        status = GroundingStatus.VERIFIED
    elif kept:
        status = GroundingStatus.PARTIAL
    else:
        status = GroundingStatus.UNVERIFIED
    return FinalizeResult(kept, status, reasons)


# --------------------------------------------------------------------------- self-tests
def _selftest() -> None:
    reg = {
        "eid_to_doc": {"ev-a": "doc1", "ev-b": "doc2"},
        "eid_to_content": {"ev-a": "The cyber sublimit is CAD 1,000,000 per claim.",
                           "ev-b": "Business interruption waiting period is 24 hours."},
    }
    # verified: excerpt contained in the SAME evidence chunk
    v = verify_reference({"evidence_id": "ev-a", "document_id": "doc1",
                          "content": "the cyber sublimit is CAD 1,000,000 per claim"}, reg)
    assert v.status == GroundingStatus.VERIFIED, v

    # excerpt not substring-confirmable but the id + document are VALID -> KEPT but only PARTIAL
    # (identity proven, SUPPORT unconfirmed -> must NOT be labeled verified). Post-review F-02.
    v2 = verify_reference({"evidence_id": "ev-a", "document_id": "doc1",
                           "content": "business interruption waiting period is 24 hours"}, reg)
    assert v2.status == GroundingStatus.PARTIAL and v2.reason == ReasonCode.EXCERPT_UNCONFIRMED, v2

    # valid id but WRONG document -> mis-attribution -> UNVERIFIED/DOCUMENT_MISMATCH (post-review F-02)
    vdm = verify_reference({"evidence_id": "ev-a", "document_id": "doc9",
                            "content": "the cyber sublimit is CAD 1,000,000 per claim"}, reg)
    assert vdm.status == GroundingStatus.UNVERIFIED and vdm.reason == ReasonCode.DOCUMENT_MISMATCH, vdm

    # hallucinated id
    v3 = verify_reference({"evidence_id": "ev-zzz", "document_id": "doc9", "content": "x" * 40}, reg)
    assert v3.reason == ReasonCode.ID_NOT_RETRIEVED, v3

    # no evidence id
    v4 = verify_reference({"evidence_id": "", "document_id": "doc1", "content": "y" * 40}, reg)
    assert v4.reason == ReasonCode.NO_EVIDENCE_ID, v4

    # finalize verified mode: one good, one hallucinated -> PARTIAL, heuristic extras dropped
    good = {"evidence_id": "ev-a", "document_id": "doc1", "content": "the cyber sublimit is CAD 1,000,000 per claim"}
    bad = {"evidence_id": "ev-zzz", "document_id": "doc9", "content": "z" * 40}
    heur = {"evidence_id": "", "document_id": "docH", "content": "heuristic extra"}
    fr = finalize([good, bad], [heur], reg, mode="verified")
    assert fr.grounding_status == GroundingStatus.PARTIAL, fr.grounding_status
    assert len(fr.references) == 1 and fr.references[0]["document_id"] == "doc1"
    assert all(r["document_id"] != "docH" for r in fr.references)  # heuristic dropped (fail-closed)

    # all verified -> VERIFIED
    fr2 = finalize([good], [], reg, mode="verified")
    assert fr2.grounding_status == GroundingStatus.VERIFIED

    # none verified though proposed -> UNVERIFIED
    fr3 = finalize([bad], [heur], reg, mode="verified")
    assert fr3.grounding_status == GroundingStatus.UNVERIFIED and fr3.references == []

    # hybrid mode -> union unchanged
    fr4 = finalize([good], [heur], reg, mode="hybrid")
    assert len(fr4.references) == 2

    print("verifier self-tests PASS")


if __name__ == "__main__":
    _selftest()
