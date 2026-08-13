"""Web search via the Tavily SDK: runs both the "general" and "news" topics
in parallel on every query, scores each response (penalizing empty or
refusal-shaped answers so a topic that actually found something always
wins over one that didn't), and returns the higher-scoring result. Also
captures structured sources (url/title/snippet/score) for citation display.

`_rehydrate_noop` is a placeholder hook for query rehydration in a
multi-tenant deployment where queries arrive PII-tokenized and need
un-tokenizing before hitting an external search API — this app has no such
pipeline (queries come straight from the Streamlit chat box, never
tokenized), so it's a pass-through.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

_WEB_PREFIX = (
    "The following is from web search, not company documents.\n\n"
)

# Default Tavily search tuning. `topic` is only a fallback used when the dual
# search below is bypassed; the primary search path runs BOTH topics explicitly
# (see `_DUAL_SEARCH_TOPICS`) and keeps the higher-scoring result, so a single
# hardcoded topic never becomes a blind spot for query types it doesn't fit
# (e.g. "news" missing anything outside its recency window).
# time_range defaults to None (no recency filter) so current-state answers older
# than a fixed window are not dropped; recency is left to Tavily's ranking. Set
# to "day"/"week"/"month"/"year" per WebSearchTool instance to bias toward fresh
# sources when desired.
_DEFAULT_SEARCH_TOPIC = "general"
_DEFAULT_SEARCH_DEPTH = "advanced"
_DEFAULT_TIME_RANGE = None
_DEFAULT_AUTO_PARAMETERS = True

# Topics searched in parallel on every query. Each is run with its topic forced
# (auto_parameters still tunes search_depth/time_range), then scored, and the
# highest-scoring response is returned. Ordered so "general" wins score ties —
# it is the safer default when both topics are equally confident.
_DUAL_SEARCH_TOPICS = ("general", "news")

# Answers that carry no real information. When Tavily's `answer` matches this on
# a short response, the topic is strongly deprioritized so a topic that actually
# answered the query wins (fixes "news" returning "I'm sorry, I don't have any
# information" while "general" answers the same query).
_REFUSAL_ANSWER_RE = re.compile(
    r"(i'?m sorry|i am sorry|i apologize|i (?:do not|don'?t) have|"
    r"no (?:relevant |useful )?information|there is no information|"
    # "the provided sources do not contain/include/mention information about ..." —
    # Tavily's standard no-answer phrasing, which the "no information" branch misses.
    r"(?:do|does|did|could|would|will)(?:n'?t| not) (?:contain|include|mention|provide|have|specify|indicate|show)|"
    r"(?:could|couldn'?t|can|can'?t|cannot|unable to)\s+(?:not\s+)?find)",
    re.IGNORECASE,
)
# A hard penalty larger than the max possible Tavily score (0..1), so any refusal
# or empty answer sorts below any genuine result regardless of its raw score.
_REFUSAL_PENALTY = 10.0


async def _rehydrate_noop(query: str, **_kwargs: Any) -> str:
    """Standalone POC replacement for
    `general_knowledge_tool.rehydrate_sanitized_query_for_agent_tool` — see
    module docstring. This POC has no PII-tokenization pipeline, so there is
    never anything to rehydrate; the query passes through unchanged."""
    return query


class WebSearchTool:
    """Web search via the Tavily SDK (``tavily-python`` ``TavilyClient``)."""

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int = 3,
        db_pool: Any | None = None,
        is_temp: bool = False,
        topic: str = _DEFAULT_SEARCH_TOPIC,
        search_depth: str = _DEFAULT_SEARCH_DEPTH,
        time_range: str | None = _DEFAULT_TIME_RANGE,
        auto_parameters: bool = _DEFAULT_AUTO_PARAMETERS,
    ):
        self.api_key = api_key
        self.max_results = max(1, min(10, max_results))
        self.pool = db_pool
        self.is_temp = is_temp
        self.topic = topic
        self.search_depth = search_depth
        self.time_range = time_range
        self.auto_parameters = auto_parameters
        self.schema_name: str | None = None
        self.thread_id: str | None = None
        self.encrypted_token: bytes | None = None
        # Web-search citations. Capture the
        # structured Tavily sources (url/title/snippet/score) for the current turn so
        # the agent can attach them as web references. Reset per turn in
        # set_rehydration_context; read via get_captured_sources() after the run.
        self._captured_sources: list[dict] = []

        self._client: Any | None
        try:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=api_key)
        except ImportError:
            self._client = None

    @staticmethod
    def _answer_from_tavily_response(data: Any) -> str:
        """Extract Tavily ``answer`` from the SDK response dict; fall back to snippets."""
        if not isinstance(data, dict):
            return (str(data) if data else "").strip()

        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        results = data.get("results")
        if isinstance(results, list) and results:
            parts: list[str] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                parts.append(f"{title}\n{content}" if title else content)
            if parts:
                return "\n\n".join(parts)

        return ""

    @staticmethod
    def _is_refusal_answer(text: str) -> bool:
        """True when an answer carries no real information (short refusal boilerplate)."""
        t = (text or "").strip()
        if not t:
            return True
        # Normalize smart quotes to straight apostrophes so a refusal written with a
        # curly quote ("I'm sorry", U+2019) is detected the same as "I'm sorry" — Tavily
        # returns the curly form, which otherwise slips past the `i'?m sorry` pattern.
        t = t.replace("’", "'").replace("‘", "'")
        # Only treat SHORT answers as refusals; a long answer that happens to
        # contain "couldn't find ..." mid-sentence is still a real answer.
        return len(t) <= 320 and bool(_REFUSAL_ANSWER_RE.search(t))

    @staticmethod
    def _max_result_score(data: Any) -> float:
        """Highest per-result relevance score (0..1) in the Tavily response."""
        if not isinstance(data, dict):
            return 0.0
        results = data.get("results")
        best = 0.0
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    score = item.get("score")
                    if isinstance(score, (int, float)):
                        best = max(best, float(score))
        return best

    def _effective_score(self, data: Any) -> float:
        """Comparable relevance for a topic: max result score, minus a hard
        penalty when the answer is empty or a refusal so relevant topics win."""
        base = self._max_result_score(data)
        raw_answer = data.get("answer") if isinstance(data, dict) else None
        usable_text = self._answer_from_tavily_response(data)
        # Prefer the model-facing answer for refusal detection; fall back to the
        # synthesized snippet text when Tavily returned no `answer` field.
        judged = raw_answer if isinstance(raw_answer, str) and raw_answer.strip() else usable_text
        if not (usable_text or "").strip() or self._is_refusal_answer(judged):
            return base - _REFUSAL_PENALTY
        return base

    def set_rehydration_context(
        self,
        *,
        schema_name: str | None = None,
        thread_id: str | None = None,
        encrypted_token: bytes | None = None,
    ) -> None:
        self.schema_name = schema_name
        self.thread_id = thread_id
        self.encrypted_token = encrypted_token
        # Reset captured web sources at the start
        # of each turn (this is called once per turn before the agent runs).
        self._captured_sources = []

    # Extract structured web sources (url/title/
    # snippet/score) from a Tavily response, deduped by url. Used for web citations.
    @staticmethod
    def _sources_from_tavily_response(data: Any, limit: int) -> list[dict]:
        if not isinstance(data, dict):
            return []
        results = data.get("results")
        if not isinstance(results, list):
            return []
        sources: list[dict] = []
        seen_urls: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            score = item.get("score")
            sources.append(
                {
                    "url": url,
                    "title": (item.get("title") or "").strip(),
                    "content": (item.get("content") or "").strip(),
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                }
            )
            if len(sources) >= max(1, limit):
                break
        return sources

    def get_captured_sources(self) -> list[dict]:
        """Structured web sources captured during this turn's search calls."""
        return list(self._captured_sources)

    def _run_tavily_search(self, query: str, topic: str | None = None) -> Any:
        """Blocking Tavily SDK call (run off the event loop via to_thread).

        `topic` overrides the instance default so the dual-topic search can force
        "general" and "news" on the same query; auto_parameters still tunes
        search_depth/time_range around the forced topic.
        """
        kwargs: dict[str, Any] = {
            "query": query,
            "topic": topic or self.topic,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": "advanced",
            "auto_parameters": self.auto_parameters,
        }
        if self.time_range:
            kwargs["time_range"] = self.time_range
        print(f"Tavily search kwargs: {kwargs}")
        return self._client.search(**kwargs)

    async def search(self, query: str) -> str:
        """Search the web for the given query."""
        if self._client is None:
            return (
                "Web search is not available: install `tavily-python`. "
                "Example: pip install tavily-python"
            )
        print(f"WebSearchTool.search called with query: {query}")
        query = await _rehydrate_noop(
            query,
            pool=self.pool,
            schema_name=self.schema_name,
            thread_id=self.thread_id,
            encrypted_token=self.encrypted_token,
            is_temp=self.is_temp,
        )
        print(f"rehydrated query: {query}")
        stripped = (query or "").strip()
        if not stripped:
            return "No search query was provided."

        # Run every topic in parallel on the same query, then keep the highest
        # scoring response. return_exceptions=True so one topic failing (or one
        # topic refusing) never sinks the whole search.
        responses = await asyncio.gather(
            *(
                asyncio.to_thread(self._run_tavily_search, stripped, topic)
                for topic in _DUAL_SEARCH_TOPICS
            ),
            return_exceptions=True,
        )

        best: tuple[float, str, str] | None = None  # (score, topic, text)
        best_response: Any | None = None  # Winning raw response, for source capture
        for topic, response in zip(_DUAL_SEARCH_TOPICS, responses):
            if isinstance(response, Exception):
                print(f"[TAVILY_ERROR] topic={topic} query={stripped!r} error={response!r}")
                continue
            print(f"Web search topic={topic} type: {type(response)}")
            print(f"Web search topic={topic} result: {response}")
            if isinstance(response, dict):
                resolved = response.get("auto_parameters")
                print(f"Tavily[{topic}] auto_parameters resolved (full): {resolved}")
                if isinstance(resolved, dict):
                    print(f"Tavily[{topic}] resolved topic: {resolved.get('topic')}")
                    print(f"Tavily[{topic}] resolved search_depth: {resolved.get('search_depth')}")
                    print(f"Tavily[{topic}] resolved time_range: {resolved.get('time_range')}")

            score = self._effective_score(response)
            text = self._answer_from_tavily_response(response)
            print(f"Tavily[{topic}] effective_score={score:.4f} has_text={bool(text.strip())}")
            # `>` (not `>=`) preserves the topic order in _DUAL_SEARCH_TOPICS as
            # the tie-break: "general" is scored first and kept on ties.
            if best is None or score > best[0]:
                best = (score, topic, text)
                best_response = response

        if best is None:
            print(f"[TAVILY_ERROR] all topics failed query={stripped!r}")
            return "Web search is temporarily unavailable. Please try again later."

        best_score, best_topic, best_text = best
        print(f"Tavily WINNER topic={best_topic} score={best_score:.4f}")

        # Capture the winning response's structured
        # sources (url/title/snippet/score) for web citations. Accumulate across any
        # multiple tavily calls in the turn; deduped by url when references are built.
        try:
            self._captured_sources.extend(
                self._sources_from_tavily_response(best_response, self.max_results)
            )
        except Exception as _src_exc:
            print(f"[TAVILY_SOURCES] capture failed: {_src_exc!r}")

        if not best_text.strip():
            return (
                _WEB_PREFIX.rstrip()
                + "Web search returned no usable text for this query."
            )
        print(f"_WEB_PREFIX + text:: {(_WEB_PREFIX + best_text)}")
        return _WEB_PREFIX + best_text
