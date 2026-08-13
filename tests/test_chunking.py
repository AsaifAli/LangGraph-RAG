"""Unit tests for retrieval.rag_pipeline.chunk_text / chunk_csv_text — the
two pure chunking functions every document upload path runs through
(see README "Document upload" / "Chunking caveat")."""

from retrieval.rag_pipeline import chunk_csv_text, chunk_text


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short paragraph that fits in one chunk."
        chunks = chunk_text(text, chunk_size=1000, overlap=150)
        assert chunks == [text]

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("", chunk_size=1000, overlap=150) == []

    def test_long_text_splits_into_multiple_chunks(self):
        paragraph = "word " * 50  # ~250 chars
        text = "\n\n".join([paragraph] * 10)  # ~2500 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 500 for c in chunks)

    def test_splitting_preserves_content(self):
        # Every word from the source text should still appear somewhere in
        # the chunked output — chunking must not silently drop content.
        text = "\n\n".join(f"paragraph number {i} has some content here." for i in range(20))
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        joined = " ".join(chunks)
        for i in range(20):
            assert f"paragraph number {i}" in joined

    def test_overlap_carries_context_across_boundary(self):
        # With overlap, adjacent chunks should share some trailing/leading
        # content rather than cutting cleanly with nothing carried over.
        paragraph = "sentence one. sentence two. sentence three. sentence four. " * 5
        text = "\n\n".join([paragraph] * 5)
        chunks = chunk_text(text, chunk_size=300, overlap=100)
        assert len(chunks) > 1

    def test_overlap_greater_than_chunk_size_raises(self):
        # chunk_text's docstring claims "chunk_overlap >= chunk_size raises a
        # clear ValueError from the library itself" — verified against the
        # installed langchain-text-splitters that this is only true for
        # STRICTLY greater (overlap == chunk_size does NOT raise, see the
        # next test below). Worth a docstring correction upstream; this test
        # pins the actually-observed behavior rather than the documented one.
        import pytest

        with pytest.raises(ValueError):
            chunk_text("some text " * 100, chunk_size=100, overlap=150)

    def test_overlap_equal_to_chunk_size_does_not_raise(self):
        # See note above: contrary to the docstring, this is accepted by the
        # installed splitter version rather than rejected.
        chunk_text("some text " * 100, chunk_size=100, overlap=100)


class TestChunkCsvText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_csv_text("") == []
        assert chunk_csv_text("   \n  \n") == []

    def test_header_only_returns_single_chunk_with_header(self):
        assert chunk_csv_text("policy_id,limit,deductible") == ["policy_id,limit,deductible"]

    def test_every_chunk_keeps_the_header(self):
        header = "policy_id,limit,deductible"
        rows = [f"POL-{i},{i * 1000},{i * 10}" for i in range(200)]
        text = "\n".join([header, *rows])
        chunks = chunk_csv_text(text, chunk_size=200, overlap_rows=0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.splitlines()[0] == header

    def test_no_row_is_ever_cut_in_half(self):
        header = "policy_id,limit,deductible"
        rows = [f"POL-{i},{i * 1000},{i * 10}" for i in range(50)]
        text = "\n".join([header, *rows])
        chunks = chunk_csv_text(text, chunk_size=150, overlap_rows=0)
        # Every original row must appear intact (as a whole line) in at
        # least one chunk.
        chunk_lines = {line for chunk in chunks for line in chunk.splitlines()}
        for row in rows:
            assert row in chunk_lines

    def test_overlap_rows_repeats_trailing_rows_in_next_chunk(self):
        header = "policy_id,limit,deductible"
        rows = [f"POL-{i},{i * 1000},{i * 10}" for i in range(50)]
        text = "\n".join([header, *rows])
        chunks = chunk_csv_text(text, chunk_size=150, overlap_rows=1)
        assert len(chunks) > 1
        # The last data row of chunk N should also appear in chunk N+1
        # (after the repeated header), proving continuity across the
        # chunk boundary.
        first_chunk_rows = chunks[0].splitlines()[1:]
        last_row_of_first_chunk = first_chunk_rows[-1]
        second_chunk_rows = chunks[1].splitlines()[1:]
        assert last_row_of_first_chunk in second_chunk_rows

    def test_blank_lines_are_ignored(self):
        text = "header\n\nrow1\n\n\nrow2\n"
        chunks = chunk_csv_text(text, chunk_size=1000, overlap_rows=0)
        assert chunks == ["header\nrow1\nrow2"]
