"""Unit tests for retrieval.rag_pipeline.content_to_text — normalizes a
LangChain message's `.content` to plain text. Exists specifically because a
real crash was found live: `ChatGoogleGenerativeAI` (the "gemini-api"
backend) returns `.content` as a list of content blocks even for a
plain-text answer, and every call site that concatenated `.content` as a
string crashed with `TypeError` the first time a Gemini turn streamed a
token (see README "A real cross-backend bug found by switching to
gemini-api")."""

from retrieval.rag_pipeline import content_to_text


def test_plain_string_passes_through():
    assert content_to_text("hello world") == "hello world"


def test_gemini_style_content_block_list():
    content = [{"type": "text", "text": "hello ", "extras": {}}, {"type": "text", "text": "world"}]
    assert content_to_text(content) == "hello world"


def test_list_of_bare_strings():
    assert content_to_text(["hello ", "world"]) == "hello world"


def test_dict_block_missing_text_key_contributes_empty_string():
    content = [{"type": "text", "text": "hello"}, {"type": "image"}]
    assert content_to_text(content) == "hello"


def test_empty_string_input():
    assert content_to_text("") == ""


def test_none_input():
    assert content_to_text(None) == ""


def test_empty_list_input():
    assert content_to_text([]) == ""
