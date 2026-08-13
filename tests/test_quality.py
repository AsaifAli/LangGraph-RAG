from shared.quality import evidence_conflict_candidates, should_abstain_from_kb


def test_abstains_when_kb_route_has_no_verified_evidence():
    assert should_abstain_from_kb(kb_requested=True, chunk_count=0, verified_count=0, web_used=False)


def test_does_not_abstain_from_web_only_turn():
    assert not should_abstain_from_kb(kb_requested=False, chunk_count=0, verified_count=0, web_used=True)


def test_detects_possible_numeric_evidence_conflict():
    answer = "The Cyber Liability sublimit is CAD 1,000,000."
    refs = [
        {"evidence_id": "E1", "content": "Cyber Liability sublimit is CAD 1,000,000 per claim."},
        {"evidence_id": "E2", "content": "Cyber Liability sublimit is CAD 2,000,000 per claim."},
    ]
    conflicts = evidence_conflict_candidates(answer, refs)
    assert conflicts
    assert set(conflicts[0]["evidence_ids"]) == {"E1", "E2"}
