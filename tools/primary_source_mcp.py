"""Katzilla MCP client adapter for EvidenceFlow.

This module intentionally speaks MCP over Streamable HTTP rather than calling
Katzilla's REST API directly. It keeps the external MCP protocol isolated from
LangGraph and normalizes Katzilla's structured result/citation envelope into a
small shape the agent can ground on.

Katzilla credentials are server-side configuration only. Never pass provider
keys or the Katzilla key through the model context.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrimarySourceResult:
    ok: bool
    agent: str
    action: str
    tool_name: str | None
    data: Any
    citation: dict[str, Any]
    quality: dict[str, Any]
    raw_text: str
    error: str | None = None


class KatzillaMCPClient:
    """Small, lazy MCP client around Katzilla's remote Streamable HTTP server."""

    def __init__(self, *, url: str, api_key: str, enabled: bool = True):
        self.url = url.rstrip('/')
        self.api_key = api_key.strip()
        self.enabled = enabled and bool(self.api_key)
        self._tool_cache: list[Any] | None = None

    @classmethod
    def from_env(cls) -> "KatzillaMCPClient":
        enabled = os.getenv("KATZILLA_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            url=os.getenv("KATZILLA_MCP_URL", "https://api.katzilla.dev/mcp"),
            api_key=os.getenv("KATZILLA_API_KEY", ""),
            enabled=enabled,
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    async def _list_tools(self, session) -> list[Any]:
        if self._tool_cache is None:
            page = await session.list_tools()
            self._tool_cache = list(page.tools)
        return self._tool_cache

    @staticmethod
    def _tool_score(tool: Any, action: str, agent: str) -> int:
        name = str(getattr(tool, "name", "") or "").lower()
        desc = str(getattr(tool, "description", "") or "").lower()
        action_l = action.lower().replace("_", "-")
        agent_l = agent.lower().replace("_", "-")
        score = 0
        if name == action_l:
            score += 100
        if action_l in name:
            score += 60
        if action_l.replace("-", "_") in name:
            score += 50
        if agent_l in name:
            score += 15
        if action_l in desc:
            score += 20
        return score

    async def _find_tool(self, session, *, action: str, agent: str) -> Any:
        tools = await self._list_tools(session)
        ranked = sorted(tools, key=lambda t: self._tool_score(t, action, agent), reverse=True)
        if not ranked or self._tool_score(ranked[0], action, agent) <= 0:
            available = ", ".join(str(getattr(t, "name", "")) for t in tools[:30])
            raise RuntimeError(
                f"Katzilla MCP action {agent}/{action!r} was not found in the advertised tool set. "
                f"Available examples: {available}"
            )
        return ranked[0]

    async def query(self, *, agent: str, action: str, params: dict[str, Any]) -> PrimarySourceResult:
        if not self.available:
            return PrimarySourceResult(
                ok=False, agent=agent, action=action, tool_name=None,
                data=None, citation={}, quality={}, raw_text="",
                error="Katzilla MCP is not configured. Set KATZILLA_ENABLED=true and KATZILLA_API_KEY.",
            )

        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            headers = {"Authorization": f"Bearer {self.api_key}"}
            timeout = httpx.Timeout(45.0, connect=10.0, read=45.0, write=15.0, pool=10.0)
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as http_client:
                async with streamable_http_client(self.url, http_client=http_client) as transport:
                    read_stream, write_stream = transport[0], transport[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tool = await self._find_tool(session, action=action, agent=agent)
                        result = await session.call_tool(str(getattr(tool, "name")), arguments=params)

                        structured = getattr(result, "structuredContent", None)
                        content_items = getattr(result, "content", None) or []
                        texts: list[str] = []
                        for item in content_items:
                            text = getattr(item, "text", None)
                            if text:
                                texts.append(str(text))
                        raw_text = "\n".join(texts)
                        payload = structured if isinstance(structured, dict) else {}
                        if not payload and raw_text:
                            try:
                                decoded = json.loads(raw_text)
                                if isinstance(decoded, dict):
                                    payload = decoded
                            except json.JSONDecodeError:
                                pass

                        citation = payload.get("citation") if isinstance(payload.get("citation"), dict) else {}
                        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
                        data = payload.get("data", payload if payload else raw_text)
                        if getattr(result, "isError", False):
                            return PrimarySourceResult(
                                ok=False, agent=agent, action=action, tool_name=str(getattr(tool, "name")),
                                data=data, citation=citation, quality=quality, raw_text=raw_text,
                                error=raw_text or "Katzilla MCP tool returned an error.",
                            )
                        return PrimarySourceResult(
                            ok=True, agent=agent, action=action, tool_name=str(getattr(tool, "name")),
                            data=data, citation=citation, quality=quality, raw_text=raw_text,
                        )
        except Exception as exc:  # noqa: BLE001 - integration boundary
            return PrimarySourceResult(
                ok=False, agent=agent, action=action, tool_name=None,
                data=None, citation={}, quality={}, raw_text="",
                error=f"Katzilla MCP request failed: {exc.__class__.__name__}: {exc}",
            )


def get_primary_source_mcp() -> KatzillaMCPClient:
    return KatzillaMCPClient.from_env()
