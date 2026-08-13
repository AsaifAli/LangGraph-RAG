"""Runtime configuration for this app: exposes only the fields this
codebase actually reads (grepped across every module, not guessed) —
Qdrant connection, the hybrid-retrieval prefetch multiplier, Tavily
web-search config, and the LibreOffice path used by the legacy-`.doc`
text-extraction fallback.

Reads this app's own `.env` (this directory) — never anything outside
`poc/langgraph_rag/`. Uses `dotenv_values` (not `load_dotenv`), same as
`rag_pipeline._get_env`, so this config never leaks into the real process
`os.environ` — see that function's docstring for why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_POC_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    """Read one config value: `.env` first, then `os.environ`, then
    `default` — same precedence as `rag_pipeline._get_env`. This module
    used to read ONLY `dotenv_values(...)`, with no `os.environ` fallback
    at all — a real bug found live in Docker: with no `.env` file in the
    image (by design, see `.dockerignore`), `QDRANT_URL` silently fell back
    straight to the hardcoded `http://localhost:6333` default, ignoring the
    `QDRANT_URL=http://qdrant:6333` docker-compose sets via `environment:`
    — every Qdrant call then tried to reach the app container's own
    localhost instead of the qdrant service, failing with `Connection
    refused` even though both containers were healthy and the network path
    was fine."""
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    legacy_key = f"POC_{key}" if not key.startswith("POC_") else key
    return (
        values.get(key)
        or os.environ.get(key)
        or values.get(legacy_key)
        or os.environ.get(legacy_key)
        or default
    )


@dataclass(frozen=True)
class VectorDBConfig:
    qdrant_url: str
    qdrant_api_key: Optional[str]


@dataclass(frozen=True)
class RerankConfig:
    hybrid_prefetch_multiplier: int


@dataclass(frozen=True)
class WebSearchConfig:
    web_search_api_key: Optional[str]
    web_search_max_results: int


@dataclass(frozen=True)
class LibConfig:
    libreoffice_path: Optional[str]


@dataclass(frozen=True)
class Config:
    vector_db: VectorDBConfig
    rerank: RerankConfig
    web_search: WebSearchConfig
    lib: LibConfig


_config: Optional[Config] = None


def load_config() -> Config:
    """Read fresh every call (not cached at import time) — matches
    `rag_pipeline.load_runtime_config`'s own freshness behavior, so editing
    `.env` takes effect on the next Streamlit rerun without a process
    restart."""
    return Config(
        vector_db=VectorDBConfig(
            qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_env("QDRANT_API_KEY", "").strip() or None,
        ),
        rerank=RerankConfig(
            hybrid_prefetch_multiplier=int(_env("HYBRID_PREFETCH_MULTIPLIER", "3")),
        ),
        web_search=WebSearchConfig(
            web_search_api_key=(
                _env("TAVILY_API_KEY", "") or _env("WEB_SEARCH_API_KEY", "")
            ).strip()
            or None,
            web_search_max_results=int(_env("WEB_SEARCH_MAX_RESULTS", "3")),
        ),
        lib=LibConfig(
            libreoffice_path=_env("LIBREOFFICE", "").strip() or None,
        ),
    )


def get_config() -> Config:
    """Cached accessor — `text_extractors.py`'s rare `.doc`-via-LibreOffice
    fallback calls this directly rather than threading a Config through
    every extractor call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
