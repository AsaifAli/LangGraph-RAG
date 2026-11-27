import io

import pytest
from openpyxl import Workbook

from agents import custom_langgraph_poc as core
from app import api_main
from shared.document_extraction import extract_document_text
from shared.upload_security import validate_document_upload


def test_public_task_update_exposes_only_approved_progress_fields():
    event = api_main._public_task_update({
        "id": "retrieve_kb-1",
        "name": "retrieve_kb",
        "status": "running",
        "prompt": "PRIVATE PROMPT",
        "internal_reasoning": "PRIVATE CHAIN OF THOUGHT",
    })
    assert event == {
        "id": "retrieve_kb-1",
        "name": "retrieve_kb",
        "label": "Search knowledge base",
        "description": "Retrieve relevant passages from the client's indexed documents.",
        "status": "running",
    }
    assert api_main._public_task_update({
        "id": "unapproved-1",
        "name": "unapproved",
        "status": "running",
    }) is None


@pytest.mark.asyncio
async def test_langgraph_stream_emits_running_and_completed_task_events(monkeypatch):
    class FakeStreamAgent:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chain_start", "name": "LangGraph",
                "run_id": "root-run", "parent_ids": [], "metadata": {},
            }
            yield {
                "event": "on_chain_start", "name": "retrieve_kb",
                "run_id": "retrieve-run", "parent_ids": ["root-run"],
                "metadata": {},
            }
            yield {
                "event": "on_chain_end", "name": "retrieve_kb",
                "run_id": "retrieve-run", "parent_ids": ["root-run"],
                "data": {"output": {"retrieved": 2}}, "metadata": {},
            }
            yield {
                "event": "on_chain_end", "name": "LangGraph",
                "run_id": "root-run", "parent_ids": [],
                "data": {"output": {"state": "complete"}}, "metadata": {},
            }

    monkeypatch.setattr(core, "validate_query", lambda query: None)
    monkeypatch.setattr(core, "make_token_tracker", lambda: object())
    monkeypatch.setattr(core, "_turn_input", lambda built, query: {"query": query})
    monkeypatch.setattr(
        core, "_finalize_outcome",
        lambda state, tracker, web_tool: {"final_answer": "verified"},
    )

    events = [
        event async for event in core.astream_langgraph_turn(
            {"agent": FakeStreamAgent(), "web_search_tool": None},
            "test question",
            thread_id="task-events-test",
        )
    ]
    task_events = [event["task"] for event in events if event.get("type") == "task"]
    assert [task["status"] for task in task_events] == ["running", "completed"]
    assert all(task["id"] == "retrieve_kb-1" for task in task_events)
    assert all(task["label"] == "Search knowledge base" for task in task_events)


def test_xlsx_upload_validation_and_extraction_preserve_sheet_and_values():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Quarterly Sales"
    sheet.append(["Region", "Revenue"])
    sheet.append(["North", 125000])
    stream = io.BytesIO()
    workbook.save(stream)
    content = stream.getvalue()

    validate_document_upload(content, "quarterly-sales.xlsx")
    extracted = extract_document_text(content, "quarterly-sales.xlsx")
    assert "Worksheet: Quarterly Sales" in extracted
    assert "Region | Revenue" in extracted
    assert "North | 125000" in extracted
