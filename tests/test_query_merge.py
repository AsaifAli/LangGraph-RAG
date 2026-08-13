"""Unit tests for shared.query_merge.merge_web_search_query — pure string
logic re-exported by retrieval.rag_pipeline.merge_web_search_query (see
that function's docstring: "Re-exported here as-is")."""

from shared.query_merge import merge_web_search_query


def test_empty_user_message_returns_tool_query():
    assert merge_web_search_query(user_message="", tool_query="Gautam Adani's son") == "Gautam Adani's son"


def test_empty_tool_query_returns_user_message():
    assert merge_web_search_query(user_message="who is his son?", tool_query="") == "who is his son?"


def test_identical_strings_return_as_is():
    q = "current Prime Minister of Nepal"
    assert merge_web_search_query(user_message=q, tool_query=q) == q


def test_short_pronoun_user_message_keeps_expanded_tool_query():
    # A short, pronoun-resolved user message with a divergent, expanded
    # tool_query: the tool_query is trusted alone (never discarded for a
    # short raw message), matching the module's own documented example.
    result = merge_web_search_query(
        user_message="his",
        tool_query="who is Gautam Adani's son",
    )
    assert result == "who is Gautam Adani's son"


def test_user_message_contains_tool_query_prefers_longer_user_message(): # noqa: E501
    result = merge_web_search_query(
        user_message="What changed in the LangGraph 1.x release notes exactly",
        tool_query="LangGraph 1.x",
    )
    assert result == "What changed in the LangGraph 1.x release notes exactly"

def test_divergent_queries_trust_the_tool_query_alone():
    # Neither phrasing contains the other -> concatenation is explicitly
    # avoided (documented as producing a run-on query that degrades
    # Tavily's relevance ranking and recency detection) -> tool_query wins.
    result = merge_web_search_query(
        user_message="who won the latest f1 race?",
        tool_query="2026 Formula 1 championship standings",
    )
    assert result == "2026 Formula 1 championship standings"


def test_whitespace_only_inputs_are_treated_as_empty():
    assert merge_web_search_query(user_message="   ", tool_query="real query") == "real query"
