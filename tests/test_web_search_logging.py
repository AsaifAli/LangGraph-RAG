import pytest
import asyncio

from tools.web_search_tool import WebSearchTool


class FakeTavily:
    def search(self, **kwargs):
        return {
            "answer": "A safe answer",
            "results": [
                {
                    "url": "https://example.com/source",
                    "title": "Example",
                    "content": "Source snippet",
                    "score": 0.9,
                }
            ],
            "auto_parameters": {"topic": kwargs["topic"]},
        }


def test_search_does_not_print_user_query_or_provider_payload(capsys):
    tool = WebSearchTool(api_key="test")
    tool._client = FakeTavily()

    result = asyncio.run(tool.search("CONFIDENTIAL QUERY DO NOT LOG"))

    captured = capsys.readouterr()
    assert "A safe answer" in result
    assert "CONFIDENTIAL QUERY DO NOT LOG" not in captured.out
    assert "Source snippet" not in captured.out
    assert tool.get_captured_sources()[0]["url"] == "https://example.com/source"


def test_web_search_availability_requires_an_initialized_sdk_client():
    tool = WebSearchTool(api_key="test")
    tool._client = None

    assert tool.available is False


@pytest.mark.asyncio
async def test_structured_search_reports_success_only_with_source_evidence():
    tool = WebSearchTool(api_key="test")
    tool._client = FakeTavily()

    result = await tool.search_structured("public safety standards")

    assert result.ok is True
    assert result.error_code is None
    assert result.sources[0]["url"] == "https://example.com/source"
    assert result.sources[0]["evidence_id"] == "W1"


@pytest.mark.asyncio
async def test_structured_search_reports_provider_failure_without_fake_answer():
    class BrokenTavily:
        def search(self, **kwargs):
            raise RuntimeError("provider unavailable")

    tool = WebSearchTool(api_key="test")
    tool._client = BrokenTavily()

    result = await tool.search_structured("public safety standards")

    assert result.ok is False
    assert result.text == ""
    assert result.sources == ()
    assert result.error_code == "provider_error"
