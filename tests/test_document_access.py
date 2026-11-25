from shared.document_access import filter_documents_for_owner, owner_scope_matches


def test_document_registry_is_scoped_to_exact_gateway_owner():
    owner_a = "a" * 64
    owner_b = "b" * 64
    records = [
        {"document_id": "doc-a", "owner_scope": owner_a},
        {"document_id": "doc-b", "owner_scope": owner_b},
        {"document_id": "legacy-doc"},
    ]

    assert [x["document_id"] for x in filter_documents_for_owner(records, owner_a)] == ["doc-a"]
    assert [x["document_id"] for x in filter_documents_for_owner(records, owner_b)] == ["doc-b"]
    assert filter_documents_for_owner(records, "c" * 64) == []


def test_document_owner_check_fails_closed_for_missing_or_invalid_identity():
    assert owner_scope_matches({"owner_scope": "a" * 64}, "a" * 64)
    assert not owner_scope_matches({"owner_scope": "a" * 64}, "b" * 64)
    assert not owner_scope_matches({"owner_scope": "a" * 64}, "")
    assert not owner_scope_matches({"document_id": "legacy"}, "a" * 64)
    assert not owner_scope_matches({"owner_scope": "legacy"}, "a" * 64)
