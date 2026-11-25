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
