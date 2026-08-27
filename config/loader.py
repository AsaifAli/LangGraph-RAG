"""Runtime configuration for the EvidenceFlow search stack.

The project no longer uses a vector database. Retrieval is implemented with
OpenSearch using classical lexical search plus optional neural-sparse search,
with the LLM used only as a targeted query planner when the first retrieval
pass is weak.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_POC_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    """Read `.env` first, then process environment, then the default."""
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    legacy_key = f"POC_{key}" if not key.startswith("POC_") else key
    return (
        os.environ.get(key)
        or os.environ.get(legacy_key)
        or values.get(key)
        or values.get(legacy_key)
        or default
    )


@dataclass(frozen=True)
class OpenSearchConfig:
    url: str
    username: Optional[str]
    password: Optional[str]
    index_name: str
    verify_certs: bool
    neural_sparse_enabled: bool
    sparse_model_name: str
    sparse_tokenizer_name: str
    sparse_prune_ratio: float
    lexical_candidate_k: int
    neural_candidate_k: int
    rerank_candidate_k: int


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool
    model_name: str


@dataclass(frozen=True)
class WebSearchConfig:
    web_search_api_key: Optional[str]
    web_search_max_results: int


@dataclass(frozen=True)
class LibConfig:
    libreoffice_path: Optional[str]


@dataclass(frozen=True)
class Config:
    opensearch: OpenSearchConfig
    rerank: RerankConfig
    web_search: WebSearchConfig
    lib: LibConfig


_config: Optional[Config] = None


def load_config() -> Config:
    return Config(
        opensearch=OpenSearchConfig(
            url=_env("OPENSEARCH_URL", "http://localhost:9200").strip(),
            username=_env("OPENSEARCH_USERNAME", "admin").strip() or None,
            password=_env("OPENSEARCH_PASSWORD", "admin").strip() or None,
            index_name=_env("OPENSEARCH_INDEX", "evidenceflow_documents").strip(),
            verify_certs=_env("OPENSEARCH_VERIFY_CERTS", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            neural_sparse_enabled=_env("OPENSEARCH_NEURAL_SPARSE", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            sparse_model_name=_env(
                "OPENSEARCH_SPARSE_MODEL",
                "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill",
            ).strip(),
            sparse_tokenizer_name=_env(
                "OPENSEARCH_SPARSE_TOKENIZER",
                "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1",
            ).strip(),
            sparse_prune_ratio=float(_env("OPENSEARCH_SPARSE_PRUNE_RATIO", "0.1")),
            lexical_candidate_k=int(_env("LEXICAL_CANDIDATE_K", "40")),
            neural_candidate_k=int(_env("NEURAL_CANDIDATE_K", "40")),
            rerank_candidate_k=int(_env("RERANK_CANDIDATE_K", "30")),
        ),
        rerank=RerankConfig(
            enabled=_env("RERANK_ENABLED", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            model_name=_env("JINA_RERANK_MODEL", _env("RERANK_MODEL", "jina-reranker-v1-turbo-en")).strip(),
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
    global _config
    if _config is None:
        _config = load_config()
    return _config
