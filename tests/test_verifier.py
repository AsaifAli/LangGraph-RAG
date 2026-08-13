"""Unit tests for citations.verifier (fully pure/stateless, no I/O). These
mirror and extend the module's own `_selftest()` (run directly via `python
citations/verifier.py`) as real pytest cases with individually reportable
failures, rather than one all-or-nothing assert chain."""

from citations.verifier import (
    GroundingStatus,
    ReasonCode,
    finalize,
    values_supported,
    verify_reference,
)

REGISTRY = {
    "eid_to_doc": {"ev-a": "doc1", "ev-b": "doc2"},
    "eid_to_content": {
        "ev-a": "The cyber sublimit is CAD 1,000,000 per claim.",
        "ev-b": "Business interruption waiting period is 24 hours.",
    },
}


class TestVerifyReference:
    def test_verified_when_excerpt_matches_its_own_evidence_chunk(self):
        v = verify_reference(
            {
                "evidence_id": "ev-a",
                "document_id": "doc1",
                "content": "the cyber sublimit is CAD 1,000,000 per claim",
            },
            REGISTRY,
        )
        assert v.status == GroundingStatus.VERIFIED
        assert v.reason == ReasonCode.OK

    def test_partial_when_excerpt_from_a_different_valid_evidence_chunk(self):
        # Identity (evidence_id -> document) is proven, but the cited text
        # doesn't come from THAT chunk -> PARTIAL, never VERIFIED (post-review
        # F-02: an unconfirmable excerpt must not be labeled verified).
        v = verify_reference(
            {
                "evidence_id": "ev-a",
                "document_id": "doc1",
                "content": "business interruption waiting period is 24 hours",
            },
            REGISTRY,
        )
        assert v.status == GroundingStatus.PARTIAL
        assert v.reason == ReasonCode.EXCERPT_UNCONFIRMED

    def test_unverified_on_document_mismatch(self):
        v = verify_reference(
            {
                "evidence_id": "ev-a",
                "document_id": "doc9",
                "content": "the cyber sublimit is CAD 1,000,000 per claim",
            },
            REGISTRY,
        )
        assert v.status == GroundingStatus.UNVERIFIED
        assert v.reason == ReasonCode.DOCUMENT_MISMATCH

    def test_unverified_on_hallucinated_evidence_id(self):
        v = verify_reference(
            {"evidence_id": "ev-zzz", "document_id": "doc9", "content": "x" * 40}, REGISTRY
        )
        assert v.status == GroundingStatus.UNVERIFIED
        assert v.reason == ReasonCode.ID_NOT_RETRIEVED

    def test_unverified_on_missing_evidence_id(self):
        v = verify_reference(
            {"evidence_id": "", "document_id": "doc1", "content": "y" * 40}, REGISTRY
        )
        assert v.status == GroundingStatus.UNVERIFIED
        assert v.reason == ReasonCode.NO_EVIDENCE_ID

    def test_short_excerpt_below_min_chars_skips_content_check(self):
        # Below min_excerpt_chars, the excerpt-match guard doesn't fire at
        # all -> falls through to VERIFIED as long as id/document line up.
        v = verify_reference(
            {"evidence_id": "ev-a", "document_id": "doc1", "content": "short"},
            REGISTRY,
            min_excerpt_chars=24,
        )
        assert v.status == GroundingStatus.VERIFIED


class TestFinalize:
    GOOD = {
        "evidence_id": "ev-a",
        "document_id": "doc1",
        "content": "the cyber sublimit is CAD 1,000,000 per claim",
    }
    BAD = {"evidence_id": "ev-zzz", "document_id": "doc9", "content": "z" * 40}
    HEURISTIC = {"evidence_id": "", "document_id": "docH", "content": "heuristic extra"}

    def test_hybrid_mode_returns_union_unchanged(self):
        result = finalize([self.GOOD], [self.HEURISTIC], REGISTRY, mode="hybrid")
        assert len(result.references) == 2
        assert result.grounding_status == GroundingStatus.NOT_APPLICABLE

    def test_verified_mode_drops_heuristics_and_hallucinations(self):
        result = finalize([self.GOOD, self.BAD], [self.HEURISTIC], REGISTRY, mode="verified")
        assert result.grounding_status == GroundingStatus.PARTIAL
        assert len(result.references) == 1
        assert result.references[0]["document_id"] == "doc1"
        assert all(r["document_id"] != "docH" for r in result.references)

    def test_verified_mode_all_good_is_fully_verified(self):
        result = finalize([self.GOOD], [], REGISTRY, mode="verified")
        assert result.grounding_status == GroundingStatus.VERIFIED

    def test_verified_mode_all_bad_is_unverified_with_empty_references(self):
        result = finalize([self.BAD], [self.HEURISTIC], REGISTRY, mode="verified")
        assert result.grounding_status == GroundingStatus.UNVERIFIED
        assert result.references == []

    def test_verified_mode_nothing_proposed_is_not_applicable(self):
        result = finalize([], [], REGISTRY, mode="verified")
        assert result.grounding_status == GroundingStatus.NOT_APPLICABLE

    def test_verified_mode_empty_registry_short_circuits(self):
        result = finalize([self.GOOD], [], {}, mode="verified")
        assert result.grounding_status == GroundingStatus.NOT_APPLICABLE
        assert result.reasons.get(ReasonCode.EMPTY_REGISTRY) == 1


class TestValuesSupported:
    def test_claim_with_no_numbers_or_dates_is_trivially_supported(self):
        ok, missing = values_supported("this is a general statement", ["some evidence"])
        assert ok is True
        assert missing == set()

    def test_claim_number_present_in_evidence_is_supported(self):
        ok, missing = values_supported(
            "the sublimit is 1000000", ["The cyber sublimit is CAD 1,000,000 per claim."]
        )
        assert ok is True

    def test_claim_number_absent_from_evidence_is_unsupported(self):
        ok, missing = values_supported("the sublimit is 999999", ["The sublimit is CAD 1,000,000."])
        assert ok is False
        assert "999999" in missing

    def test_claim_date_present_in_evidence_is_supported(self):
        ok, _ = values_supported(
            "the policy starts 2025-01-01", ["Policy period runs from 2025-01-01 to 2026-01-01."]
        )
        assert ok is True
