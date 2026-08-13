"""Unit tests for agents.custom_langgraph_poc._resolve_full_fetch_targets —
which document(s) a SUMMARY/COMPARE-mode turn fetches in full. Exists
specifically because a real bug was found live: asking to summarize a
named file that doesn't match anything in scope ("summarize DON.txt" with
only unrelated documents actually uploaded) used to silently fall back to
fetching EVERY document in scope, producing an answer that summarized
unrelated documents instead of saying the named file doesn't exist."""

from agents.custom_langgraph_poc import _resolve_full_fetch_targets

NAMES = {
    "id1": "Cornerstone Distributing Co.txt",
    "id2": "Chaman lal corporation limited.pdf",
}
IDS = list(NAMES.keys())


def test_single_document_in_scope_always_returns_it():
    # len(document_ids) <= 1 short-circuits before any name matching.
    assert _resolve_full_fetch_targets("summarize this", ["id1"], NAMES) == ["id1"]
    assert _resolve_full_fetch_targets("summarize DON.txt", ["id1"], NAMES) == ["id1"]


def test_real_matching_name_resolves_to_that_document():
    result = _resolve_full_fetch_targets(
        "summarize Cornerstone Distributing Co.txt", IDS, NAMES
    )
    assert result == ["id1"]


def test_two_matching_names_resolve_to_both_for_compare():
    result = _resolve_full_fetch_targets(
        "compare Cornerstone Distributing Co.txt and Chaman lal corporation limited.pdf",
        IDS,
        NAMES,
    )
    assert set(result) == {"id1", "id2"}


def test_vague_summarize_everything_falls_back_to_all():
    result = _resolve_full_fetch_targets("summarize all my documents", IDS, NAMES)
    assert set(result) == set(IDS)


def test_no_specific_name_falls_back_to_all():
    result = _resolve_full_fetch_targets("what topics does this cover", IDS, NAMES)
    assert set(result) == set(IDS)


def test_unmatched_specific_filename_returns_empty_not_everything():
    # The actual bug: a query names a real-looking file that isn't in
    # scope. Must return [] (signals "not found" downstream via empty
    # chunks + SYNTHESIS_PROMPT's own "say so plainly" rule), NOT silently
    # fall back to summarizing whatever unrelated documents are in scope.
    result = _resolve_full_fetch_targets("summarize DON.txt", IDS, NAMES)
    assert result == []


def test_unmatched_filename_with_different_extension_also_returns_empty():
    result = _resolve_full_fetch_targets("what does report.pdf say", IDS, NAMES)
    assert result == []


def test_unmatched_doc_and_xlsx_also_return_empty():
    # .doc (via the `docx?` alternation) and .xlsx (not an ingestible type
    # in this app at all, see app/streamlit_app.py's file_uploader — but
    # a user can still type "summarize sales.xlsx" for one that was never
    # uploaded, and that should resolve the same way as any other
    # unmatched filename-shaped reference).
    assert _resolve_full_fetch_targets("summarize notes.doc", IDS, NAMES) == []
    assert _resolve_full_fetch_targets("summarize sales.xlsx", IDS, NAMES) == []
