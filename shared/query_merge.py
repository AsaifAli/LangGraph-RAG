"""`merge_web_search_query` decides the effective web-search query when an
agent has both the raw user message and its own rewritten `tool_query` for
the same turn. Pure string logic, zero dependencies."""


def merge_web_search_query(*, user_message: str, tool_query: str) -> str:
    """Build effective web-search query: never discard the agent's tool_query.

    Unlike merge_tool_query, a short tool_query is not dropped in favor of the
    raw user message — a short tool_query is often the agent's pronoun/context
    resolution (e.g. "his" -> "Gautam Adani's son"), and discarding it forces
    an ambiguous query into search.

    When the two phrasings diverge (neither contains the other), the agent's
    tool_query is trusted alone rather than concatenated with the raw message.
    Concatenation produces a duplicated, run-on query (e.g. "latest f1 race
    winner?\n\nWho won the latest f1 race?") that degrades Tavily's relevance
    ranking and has been observed to skew its auto-parameter recency detection
    (misreading "latest" as same-day and dropping the actual answer). The tool
    query is expected to already be a complete, self-contained question per
    the tool's own instructions, so it stands on its own.
    """
    user = (user_message or "").strip()
    tool = (tool_query or "").strip()
    if not user:
        return tool
    if not tool:
        return user
    if user == tool:
        return user
    if user.lower() in tool.lower() or tool.lower() in user.lower():
        return tool if len(tool) >= len(user) else user
    return tool
