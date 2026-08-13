"""Data analysis pattern — narrow, single-script, dev-only.
Reference: https://docs.langchain.com/oss/python/deepagents/data-analysis

Chunking in this app is a simple recursive-splitter (`rag_pipeline.chunk_text`/
`chunk_csv_text`) — not a structure-aware chunking service, so this script
exists to compare two ways of answering a tabular aggregation question:

  1. Points a dev-only LocalShellBackend deepagents instance at a small
     synthetic fixture (fixtures/sample_policy_schedule.csv) and asks it a
     cross-row AGGREGATION question pandas can answer directly
     (`df.groupby(...).sum()`).
  2. Asks the SAME question of this app's own retrieval path
     (`rag_pipeline.retrieve_and_rerank`) against the demo KB seeded by
     seed_demo_kb.py, to see whether row-level semantic-chunk retrieval can
     answer an aggregation query at all.

Goal is narrow: is a code-interpreter agent meaningfully better at
structured/tabular QA than semantic chunk retrieval, or redundant with it?
Not a general-purpose data analysis feature.

LocalShellBackend runs arbitrary shell/python in a local subprocess —
DEV-ONLY, never point this at anything other than the bundled fixture.

Run (from poc/langgraph_rag/): python scripts/data_analysis_poc.py [--backend gemini-api|huggingface]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# This file lives in scripts/ — POC root is one level up, needed so
# `python scripts/data_analysis_poc.py` resolves `from retrieval...` below
# regardless of CWD.
_POC_ROOT = Path(__file__).resolve().parent.parent
if str(_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_POC_ROOT))

from retrieval.rag_pipeline import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_FIXTURE_RELATIVE_PATH,
    DEFAULT_TOP_K,
    PLATFORM_TENANT_ID,
    TENANT_SCHEMA,
    build_langchain_model,
    content_to_text,
    make_token_tracker,
    retrieve_and_rerank,
    summarize_token_usage,
)

_FIXTURE = _POC_ROOT / DEFAULT_FIXTURE_RELATIVE_PATH
_QUESTION = (
    "Across all policies underwritten by Northbridge Underwriters, what is the "
    "total Cyber Liability sublimit in CAD?"
)

AGENT_INSTRUCTIONS = f"""You are a data-analysis agent with shell/code execution access.

A CSV file is available at /data/sample_policy_schedule.csv with columns:
policy_number, coverage_type, sublimit_cad, deductible_cad, effective_date,
expiry_date, carrier.

Answer this question using pandas (read the file, filter, aggregate):
{_QUESTION}

Show your work (the code you ran) and state the final numeric answer clearly.
"""


async def run_code_interpreter_agent(backend: str) -> dict[str, Any]:
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    fs_backend = LocalShellBackend(
        root_dir=str(_FIXTURE.parent),
        virtual_mode=True,
    )
    fs_backend.upload_files(
        [("/data/sample_policy_schedule.csv", _FIXTURE.read_bytes())]
    )

    model = build_langchain_model(backend)
    agent = create_deep_agent(model=model, backend=fs_backend, system_prompt=AGENT_INSTRUCTIONS)
    token_tracker = make_token_tracker()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": _QUESTION}]},
        config={"callbacks": [token_tracker]},
    )
    return {
        # See rag_pipeline.content_to_text — the default "gemini-api" backend
        # returns list-shaped content, which `run` would otherwise print as a
        # raw list of block dicts instead of the answer.
        "answer": content_to_text(result["messages"][-1].content),
        "token_usage": summarize_token_usage(token_tracker),
    }


async def run_kb_retrieval_baseline() -> str:
    """What this app's own retrieval path returns for the SAME question,
    against the demo KB (see seed_demo_kb.py) — the "chunking + KB search"
    baseline to compare the code-interpreter answer against."""
    chunks = await retrieve_and_rerank(
        _QUESTION,
        tenant_schema=TENANT_SCHEMA,
        platform_tenant_id=PLATFORM_TENANT_ID,
        document_ids=["11111111-1111-1111-1111-111111111111"],
        top_k=DEFAULT_TOP_K,
    )
    if not chunks:
        return "(no chunks retrieved — see README on seeding demo data with seed_demo_kb.py)"
    return "\n".join(f"[{c.evidence_id} score={c.score:.3f}] {c.content}" for c in chunks)


async def run_data_analysis(*, backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Run both halves of the comparison and return structured results — the
    shared core both `main()` (CLI) and streamlit_app.py call."""
    outcome: dict[str, Any] = {"question": _QUESTION}

    try:
        ci_result = await run_code_interpreter_agent(backend)
        outcome["code_interpreter_answer"] = ci_result["answer"]
        outcome["code_interpreter_token_usage"] = ci_result["token_usage"]
    except Exception as exc:  # noqa: BLE001 - POC comparison, report and continue
        outcome["code_interpreter_error"] = repr(exc)

    try:
        outcome["kb_baseline"] = await run_kb_retrieval_baseline()
    except Exception as exc:  # noqa: BLE001
        outcome["kb_baseline_error"] = repr(exc)

    return outcome


async def run(args: argparse.Namespace) -> None:
    outcome = await run_data_analysis(backend=args.backend)

    print(f"Question: {outcome['question']}\n")

    print("=== Code-interpreter agent (deepagents + LocalShellBackend) ===")
    if "code_interpreter_answer" in outcome:
        print(outcome["code_interpreter_answer"])
        usage = outcome["code_interpreter_token_usage"]
        print(f"\n[tokens] total={usage['total_tokens']} input={usage['input_tokens']} output={usage['output_tokens']}")
    else:
        print(f"[data_analysis_poc] code-interpreter run failed: {outcome['code_interpreter_error']}")

    print("\n=== KB semantic-chunk retrieval baseline (retrieve_and_rerank) ===")
    if "kb_baseline" in outcome:
        print(outcome["kb_baseline"])
    else:
        print(f"[data_analysis_poc] KB baseline failed: {outcome['kb_baseline_error']}")

    print(
        "\nExpected finding: semantic chunk retrieval returns the most similar "
        "individual row-level chunks (relevant but not summed), while the "
        "code-interpreter agent can compute the actual cross-row total — the "
        "gap to weigh before relying on chunk retrieval alone for tabular "
        "aggregation questions over real documents."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["gemini-api", "huggingface"], default=DEFAULT_BACKEND
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
