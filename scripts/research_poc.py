"""Deep research / web search pattern.
Reference: https://docs.langchain.com/oss/python/deepagents/deep-research

Wraps Tavily-backed web search (`tools/web_search_tool.py::WebSearchTool`)
and a deterministic query-merge step (`merge_web_search_query` — pure
string logic, not an LLM rewrite: it prefers the longer/more specific of
the agent's tool_query vs. the user's raw message) as the tool for a
deepagents `research-agent` subagent, with `write_todos` (via
`TodoListMiddleware`) for planning and a delegation-budget pattern
(`max_concurrent_research_units`, `max_researcher_iterations`) matching the
docs.

Two chat-model backends are available — see rag_pipeline.build_langchain_model:
"huggingface" (Qwen2.5-72B-Instruct via HF Inference API, the default) and
"gemini-api" (Gemini over a direct Google AI Studio API key). `--backend all`
runs both and reports latency/result differences.

Run (from poc/langgraph_rag/): python scripts/research_poc.py --query "..." [--backend gemini-api|huggingface|all]
     [--max-concurrent-research-units 3] [--max-researcher-iterations 3]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# This file lives in scripts/ — POC root is one level up, needed so
# `python scripts/research_poc.py` resolves `from retrieval...` below
# regardless of CWD.
_POC_ROOT = Path(__file__).resolve().parent.parent
if str(_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_POC_ROOT))

from retrieval.rag_pipeline import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    DEFAULT_MAX_RESEARCHER_ITERATIONS,
    build_langchain_model,
    content_to_text,
    get_web_search_tool,
    make_token_tracker,
    merge_web_search_query,
    summarize_token_usage,
)

ORCHESTRATOR_INSTRUCTIONS = """You are a research orchestrator (deepagents-based).

For the user's question:
1. Use `write_todos` to plan your research (1-3 short steps).
2. Delegate to the `research-agent` subagent via task(). Use at most
   {max_concurrent_research_units} parallel research-agent delegations, and
   stop delegating after {max_researcher_iterations} rounds even if sources
   feel incomplete — report what you have instead of looping indefinitely.
3. Synthesize a final answer from the research-agent's findings, noting
   which claims came from web search vs. general knowledge.
"""

RESEARCHER_INSTRUCTIONS = """You are a research-agent subagent with access to `tavily_search`.

Search the web for the given topic. Stop after at most 5 `tavily_search`
calls even if you cannot find a good source, and stop as soon as you have
gathered enough information to answer the topic confidently. Report your
findings in 2-4 sentences, noting source recency/quality where relevant.
"""


def _make_tavily_tool():
    from langchain_core.tools import tool

    web_search_tool = get_web_search_tool()

    @tool(parse_docstring=True)
    async def tavily_search(query: str) -> str:
        """Search the web via Tavily for current information.

        Args:
            query: The search query.
        """
        # Same merge step conversation_agent.py runs before every tavily_search
        # call — see module docstring for why this is a deterministic merge,
        # not an LLM rewrite.
        effective_query = merge_web_search_query(user_message=query, tool_query=query)
        return await web_search_tool.search(effective_query)

    return tavily_search


async def run_one_backend(args: argparse.Namespace, backend: str) -> dict:
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware

    tavily_tool = _make_tavily_tool()
    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent. Give one topic at a time.",
        "system_prompt": RESEARCHER_INSTRUCTIONS,
        "tools": [tavily_tool],
    }

    model = build_langchain_model(backend)
    agent = create_deep_agent(
        model=model,
        tools=[tavily_tool],
        system_prompt=ORCHESTRATOR_INSTRUCTIONS.format(
            max_concurrent_research_units=args.max_concurrent_research_units,
            max_researcher_iterations=args.max_researcher_iterations,
        ),
        subagents=[research_sub_agent],
        middleware=[TodoListMiddleware()],
    )

    token_tracker = make_token_tracker()
    start = time.monotonic()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": args.query}]},
        config={"callbacks": [token_tracker]},
    )
    elapsed = time.monotonic() - start
    # See rag_pipeline.content_to_text — the default "gemini-api" backend
    # returns list-shaped content, which would otherwise be printed as a raw
    # list of block dicts by the comparison table in `run`.
    final_message = content_to_text(result["messages"][-1].content)
    tool_call_count = sum(
        len(getattr(m, "tool_calls", None) or []) for m in result["messages"]
    )
    return {
        "backend": backend,
        "elapsed_seconds": elapsed,
        "tool_call_count": tool_call_count,
        "final_answer": final_message,
        "token_usage": summarize_token_usage(token_tracker),
    }


async def run_research(query: str, *, backend: str = DEFAULT_BACKEND,
                        max_concurrent_research_units: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
                        max_researcher_iterations: int = DEFAULT_MAX_RESEARCHER_ITERATIONS) -> list[dict]:
    """Run the deep-research pattern against one or all backends and return
    structured results — the shared core both `main()` (CLI) and
    streamlit_app.py call."""
    args = argparse.Namespace(
        query=query,
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
    backends = ["gemini-api", "huggingface"] if backend == "all" else [backend]
    results = []
    for b in backends:
        try:
            outcome = await run_one_backend(args, b)
        except Exception as exc:  # noqa: BLE001 - POC comparison, report and continue
            outcome = {"backend": b, "error": repr(exc)}
        results.append(outcome)
    return results


async def run(args: argparse.Namespace) -> None:
    results = await run_research(
        args.query,
        backend=args.backend,
        max_concurrent_research_units=args.max_concurrent_research_units,
        max_researcher_iterations=args.max_researcher_iterations,
    )

    for r in results:
        if "error" in r:
            print(f"\n=== Running backend: {r['backend']} ===", file=sys.stderr)
            print(f"[research_poc] backend={r['backend']} failed: {r['error']}", file=sys.stderr)

    print("\n=== CROSS-BACKEND COMPARISON ===\n")
    for r in results:
        if "error" in r:
            print(f"{r['backend']:>7}: FAILED — {r['error']}")
            continue
        usage = r["token_usage"]
        print(
            f"{r['backend']:>7}: {r['elapsed_seconds']:.1f}s, "
            f"{r['tool_call_count']} tool calls, {usage['total_tokens']} total tokens"
        )
        print(f"          answer: {r['final_answer'][:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--backend",
        choices=["gemini-api", "huggingface", "all"],
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--max-concurrent-research-units", type=int, default=DEFAULT_MAX_CONCURRENT_SUBAGENTS
    )
    parser.add_argument(
        "--max-researcher-iterations", type=int, default=DEFAULT_MAX_RESEARCHER_ITERATIONS
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
