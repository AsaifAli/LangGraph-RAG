"""EvidenceFlow retrieval and ingestion.

Retrieval is intentionally *sparse-first* and contains no dense-vector
search or vector database:

1. Classical OpenSearch lexical retrieval (BM25, exact, phrase, fuzzy,
   field boosting and metadata filters).
2. Optional OpenSearch neural-sparse retrieval. OpenSearch stores sparse
   token-weight representations in an inverted-index-oriented structure;
   there is no k-NN dense vector search in this project.
3. Client-side reciprocal-rank fusion (RRF) across lexical and neural-sparse
   candidates.
4. Cross-encoder reranking via the configured backend: Jina's hosted
   reranker API for Render, with an optional local FastEmbed implementation
   for development/offline use.
5. Agentic query expansion only when the first pass looks weak. The agent
   generates a few alternative lexical queries; OpenSearch still performs
   the actual retrieval.

The public functions keep the API expected by the existing LangGraph graph,
Streamlit app and CLI scripts so the application can be re-engineered without
rewriting unrelated layers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_POC_DIR = Path(__file__).resolve().parent.parent


def _get_env(key: str, default: str) -> str:
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


from shared.constants import AIConstants

TENANT_SCHEMA = _get_env("TENANT_SCHEMA", "poc_demo")
PLATFORM_TENANT_ID = _get_env("PLATFORM_TENANT_ID", "langgraph_rag")
DEFAULT_BACKEND = _get_env("BACKEND", "litellm").strip().lower()
DEFAULT_TOP_K = int(_get_env("TOP_K", "5"))
DEFAULT_MAX_CONCURRENT_SUBAGENTS = int(_get_env("MAX_CONCURRENT_SUBAGENTS", "3"))
DEFAULT_MAX_RESEARCHER_ITERATIONS = int(_get_env("MAX_RESEARCHER_ITERATIONS", "3"))
DEFAULT_CHUNK_SIZE = int(_get_env("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", "150"))
DEFAULT_CSV_OVERLAP_ROWS = int(_get_env("CSV_OVERLAP_ROWS", "1"))
DEFAULT_WEB_SEARCH_MAX_RESULTS = int(_get_env("WEB_SEARCH_MAX_RESULTS", "3"))
DEFAULT_MAX_WEB_SEARCH_CALLS = int(_get_env("MAX_WEB_SEARCH_CALLS", "5"))
DEFAULT_MAX_SUMMARY_CHUNKS = int(_get_env("MAX_SUMMARY_CHUNKS", "40"))
DEFAULT_TURN_TIMEOUT_SECONDS = int(_get_env("TURN_TIMEOUT_SECONDS", "180"))
DEFAULT_FIXTURE_RELATIVE_PATH = _get_env("FIXTURE_PATH", "fixtures/sample_policy_schedule.csv")

# OpenSearch retrieval settings.
DEFAULT_OPENSEARCH_INDEX = _get_env("OPENSEARCH_INDEX", "evidenceflow_documents")
OPENSEARCH_URL = _get_env("OPENSEARCH_URL", "http://localhost:9200").strip()
OPENSEARCH_USERNAME = _get_env("OPENSEARCH_USERNAME", "admin").strip()
OPENSEARCH_PASSWORD = _get_env("OPENSEARCH_PASSWORD", "admin").strip()
OPENSEARCH_VERIFY_CERTS = _get_env("OPENSEARCH_VERIFY_CERTS", "false").strip().lower() in {"1", "true", "yes", "on"}
OPENSEARCH_NEURAL_SPARSE = _get_env("OPENSEARCH_NEURAL_SPARSE", "true").strip().lower() in {"1", "true", "yes", "on"}
OPENSEARCH_SPARSE_MODEL = _get_env(
    "OPENSEARCH_SPARSE_MODEL",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill",
)
OPENSEARCH_SPARSE_TOKENIZER = _get_env(
    "OPENSEARCH_SPARSE_TOKENIZER",
    "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1",
)
OPENSEARCH_SPARSE_PRUNE_RATIO = float(_get_env("OPENSEARCH_SPARSE_PRUNE_RATIO", "0.1"))
LEXICAL_CANDIDATE_K = int(_get_env("LEXICAL_CANDIDATE_K", "40"))
NEURAL_CANDIDATE_K = int(_get_env("NEURAL_CANDIDATE_K", "40"))
RERANK_CANDIDATE_K = int(_get_env("RERANK_CANDIDATE_K", "30"))
RERANK_ENABLED = _get_env("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RERANK_BACKEND = _get_env("RERANK_BACKEND", "local").strip().lower()
DEFAULT_RERANK_MODEL = _get_env("RERANK_MODEL", "jina-reranker-v1-turbo-en")
JINA_API_KEY = _get_env("JINA_API_KEY", "").strip()
JINA_API_URL = _get_env("JINA_API_URL", "https://api.jina.ai/v1/rerank").strip()
JINA_RERANK_MODEL = _get_env("JINA_RERANK_MODEL", "jina-reranker-v1-turbo-en").strip()
AGENTIC_RETRIEVAL_ENABLED = _get_env("AGENTIC_RETRIEVAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_AGENTIC_RETRIEVAL_ROUNDS = int(_get_env("MAX_AGENTIC_RETRIEVAL_ROUNDS", "2"))
AGENTIC_RETRIEVAL_MIN_HITS = int(_get_env("AGENTIC_RETRIEVAL_MIN_HITS", "2"))
RRF_K = int(_get_env("RRF_K", "60"))

# Populated lazily after OpenSearch deploys the local sparse models.
_SPARSE_ENCODING_MODEL_ID: str | None = None
_SPARSE_TOKENIZER_MODEL_ID: str | None = None

HF_CHAT_MODEL = _get_env("HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
GEMINI_API_MODEL = _get_env("GEMINI_API_MODEL", "gemini-flash-lite-latest")


async def _to_thread_with_retry(func: Any, *args: Any, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def get_hf_token() -> str:
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    token = values.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN not found in {_POC_DIR / '.env'}")
    return token


def get_gemini_api_key() -> str:
    from dotenv import dotenv_values

    values = dotenv_values(_POC_DIR / ".env")
    key = values.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(f"GEMINI_API_KEY not found in {_POC_DIR / '.env'}")
    return key


def load_runtime_config():
    from config.loader import load_config

    return load_config()


@dataclass
class RetrievedChunk:
    evidence_id: str
    document_id: str
    content: str
    score: float
    meta_data: dict


def _safe_index_name(tenant_schema: str, platform_tenant_id: str) -> str:
    raw = f"{tenant_schema}_{platform_tenant_id}_{DEFAULT_OPENSEARCH_INDEX}"
    safe = re.sub(r"[^a-z0-9._-]+", "-", raw.lower()).strip("-_")
    return safe[:200]


def opensearch_index_name(tenant_schema: str, platform_tenant_id: str) -> str:
    return _safe_index_name(tenant_schema, platform_tenant_id)


def build_opensearch_client():
    from opensearchpy import OpenSearch
    from urllib.parse import urlparse

    parsed = urlparse(OPENSEARCH_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 9200)
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
        use_ssl=parsed.scheme == "https",
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
        timeout=20,
        max_retries=2,
        retry_on_timeout=True,
        http_compress=True,
    )


def _perform_plugin_request(client: Any, method: str, path: str, *, body: dict | None = None) -> dict:
    response = client.transport.perform_request(method, path, body=body)
    if isinstance(response, dict):
        return response
    return json.loads(response) if response else {}


def _find_model_id(client: Any, model_name: str) -> str | None:
    try:
        result = _perform_plugin_request(
            client,
            "POST",
            "/_plugins/_ml/models/_search",
            body={"query": {"match": {"name": model_name}}, "size": 10},
        )
    except Exception:
        return None
    hits = (((result.get("hits") or {}).get("hits")) or [])
    for hit in hits:
        source = hit.get("_source") or {}
        candidate = source.get("model_id") or hit.get("_id")
        if candidate and source.get("name") == model_name:
            return str(candidate)
    return None


def _register_and_deploy_model(client: Any, model_name: str, version: str = "1.0.0") -> str:
    existing = _find_model_id(client, model_name)
    if existing:
        return existing

    response = _perform_plugin_request(
        client,
        "POST",
        "/_plugins/_ml/models/_register?deploy=true",
        body={"name": model_name, "version": version, "model_format": "TORCH_SCRIPT"},
    )
    task_id = response.get("task_id")
    if not task_id:
        raise RuntimeError(f"OpenSearch model registration did not return task_id: {response}")

    deadline = time.time() + 300
    while time.time() < deadline:
        task = _perform_plugin_request(client, "GET", f"/_plugins/_ml/tasks/{task_id}")
        state = str(task.get("state") or task.get("status") or "").upper()
        if state == "COMPLETED":
            model_id = task.get("model_id") or ((task.get("result") or {}).get("model_id"))
            if model_id:
                return str(model_id)
        if state in {"FAILED", "CANCELLED", "ERROR"}:
            raise RuntimeError(f"OpenSearch model deployment failed: {task}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for OpenSearch model deployment: {model_name}")


def ensure_opensearch_index(client: Any, *, index_name: str) -> None:
    """Create the lexical + optional neural-sparse index and ingestion pipeline.

    Neural sparse is additive. If local model deployment is unavailable, lexical
    retrieval continues to work. The model IDs are cached process-wide once the
    OpenSearch ML plugin has registered/deployed them.
    """
    global _SPARSE_ENCODING_MODEL_ID, _SPARSE_TOKENIZER_MODEL_ID

    sparse_enabled = bool(OPENSEARCH_NEURAL_SPARSE)
    if sparse_enabled:
        try:
            client.cluster.put_settings(
                body={
                    "persistent": {
                        "plugins.ml_commons.only_run_on_ml_node": False,
                        "plugins.ml_commons.model_access_control_enabled": True,
                        "plugins.ml_commons.native_memory_threshold": 99,
                    }
                }
            )
        except Exception:
            # A restricted OpenSearch deployment may not allow cluster-setting
            # updates. The retrieval path remains usable; model deployment below
            # will fail closed to lexical-only mode if the cluster disallows it.
            pass
    sparse_model_id = _SPARSE_ENCODING_MODEL_ID
    tokenizer_model_id = _SPARSE_TOKENIZER_MODEL_ID
    if sparse_enabled and (not sparse_model_id or not tokenizer_model_id):
        try:
            sparse_model_id = sparse_model_id or _register_and_deploy_model(client, OPENSEARCH_SPARSE_MODEL)
            tokenizer_model_id = tokenizer_model_id or _register_and_deploy_model(client, OPENSEARCH_SPARSE_TOKENIZER, version="1.0.1")
            _SPARSE_ENCODING_MODEL_ID = sparse_model_id
            _SPARSE_TOKENIZER_MODEL_ID = tokenizer_model_id
        except Exception:
            sparse_enabled = False

    index_exists = client.indices.exists(index=index_name)
    if index_exists:
        if sparse_enabled:
            try:
                mapping = client.indices.get_mapping(index=index_name)
                props = ((mapping.get(index_name) or {}).get("mappings") or {}).get("properties") or {}
                sparse_enabled = "passage_embedding" in props
            except Exception:
                sparse_enabled = False
        return

    mapping: dict[str, Any] = {
        "dynamic": "false",
        "properties": {
            "passage_text": {"type": "text"},
            "title": {"type": "text"},
            "heading": {"type": "text"},
            "section": {"type": "text"},
            "keywords": {"type": "text"},
            "document_id": {"type": "keyword"},
            "tenant_schema": {"type": "keyword"},
            "platform_tenant_id": {"type": "keyword"},
            "source_name": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "chunk_number": {"type": "integer"},
            "page_no": {"type": "integer"},
            "thread_id": {"type": "keyword"},
            "person_ids": {"type": "keyword"},
            "org_ids": {"type": "keyword"},
            "created_at": {"type": "date"},
        },
    }
    if sparse_enabled:
        mapping["properties"]["passage_embedding"] = {"type": "rank_features"}

    body: dict[str, Any] = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping}

    if sparse_enabled and sparse_model_id:
        pipeline_name = f"{index_name}_neural_sparse"
        client.ingest.put_pipeline(
            id=pipeline_name,
            body={
                "description": "EvidenceFlow neural sparse ingestion",
                "processors": [
                    {
                        "sparse_encoding": {
                            "model_id": sparse_model_id,
                            "prune_type": "max_ratio",
                            "prune_ratio": OPENSEARCH_SPARSE_PRUNE_RATIO,
                            "field_map": {"passage_text": "passage_embedding"},
                        }
                    }
                ],
            },
        )
        body["settings"]["default_pipeline"] = pipeline_name

    client.indices.create(index=index_name, body=body)


def _flatten_metadata(
    *, document_id: str, tenant_schema: str, platform_tenant_id: str, source_name: str,
    chunk_number: int, content: str
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "tenant_schema": f"{tenant_schema}_{platform_tenant_id}",
        "platform_tenant_id": platform_tenant_id,
        "source_name": source_name,
        "chunk_id": f"{document_id}:{chunk_number}",
        "chunk_number": chunk_number,
        "title": Path(source_name).stem,
        "heading": "",
        "section": source_name,
        "keywords": "",
        "page_no": None,
        "passage_text": content,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _source_to_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    source = hit.get("_source") or {}
    score = float(hit.get("_score") or 0.0)
    document_id = str(source.get("document_id") or "")
    chunk_number = int(source.get("chunk_number") or 0)
    meta = {
        "document_id": document_id,
        "tenant_schema": source.get("tenant_schema", ""),
        "platform_tenant_id": source.get("platform_tenant_id", ""),
        "source_name": source.get("source_name", ""),
        "chunk_id": source.get("chunk_id", ""),
        "chunk_number": chunk_number,
        "section_path": source.get("section") or source.get("source_name", ""),
        "page_no": source.get("page_no"),
        "title": source.get("title", ""),
        "heading": source.get("heading", ""),
        "keywords": source.get("keywords", ""),
    }
    evidence_id = f"E{chunk_number or abs(hash((document_id, source.get('chunk_id')))) % 100000}"
    return RetrievedChunk(
        evidence_id=evidence_id,
        document_id=document_id,
        content=str(source.get("passage_text") or ""),
        score=score,
        meta_data=meta,
    )


def chunk_text(text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def chunk_csv_text(text: str, *, max_rows: int = 24, overlap_rows: int = DEFAULT_CSV_OVERLAP_ROWS) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0]
    rows = lines[1:]
    chunks: list[str] = []
    step = max(1, max_rows - overlap_rows)
    for start in range(0, len(rows), step):
        window = rows[start : start + max_rows]
        if window:
            chunks.append("\n".join([header, *window]))
    return chunks


def extract_text(content: bytes, source_name: str) -> str:
    from shared.text_extractors import DocumentTextExtractor

    extractor = DocumentTextExtractor()
    suffix = Path(source_name).suffix.lower().lstrip(".")
    if suffix == "pdf":
        return extractor.pdf_bytes_to_text(content)
    if suffix == "docx":
        return extractor.docx_bytes_to_text(content)
    if suffix == "doc":
        return extractor.doc_bytes_to_text(content)
    return extractor.txt_bytes_to_text(content)


async def ingest_document(
    content: bytes,
    *,
    source_name: str,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    document_id: str | None = None,
) -> dict:
    if document_id is None:
        document_id = str(uuid.uuid4())

    text = extract_text(content, source_name)
    chunks = chunk_csv_text(text) if source_name.lower().endswith(".csv") else chunk_text(text)
    if not chunks:
        return {"document_id": document_id, "chunk_count": 0}

    client = build_opensearch_client()
    index_name = opensearch_index_name(tenant_schema, platform_tenant_id)
    await _to_thread_with_retry(ensure_opensearch_index, client, index_name=index_name)

    actions = []
    for i, chunk in enumerate(chunks, start=1):
        doc = _flatten_metadata(
            document_id=document_id,
            tenant_schema=tenant_schema,
            platform_tenant_id=platform_tenant_id,
            source_name=source_name,
            chunk_number=i,
            content=chunk,
        )
        actions.append({"index": {"_index": index_name, "_id": f"{document_id}:{i}"}})
        actions.append(doc)

    def _bulk() -> Any:
        return client.bulk(body=actions, refresh="wait_for")

    response = await _to_thread_with_retry(_bulk)
    if response.get("errors"):
        failures = [item for item in response.get("items", []) if item.get("index", {}).get("error")]
        raise RuntimeError(f"OpenSearch ingestion failed for {len(failures)} chunks: {failures[:2]}")
    return {"document_id": document_id, "chunk_count": len(chunks)}


_RERANK_ENCODER_CACHE: dict[str, Any] = {}


def _get_rerank_encoder(model_name: str):
    encoder = _RERANK_ENCODER_CACHE.get(model_name)
    if encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        encoder = TextCrossEncoder(model_name=model_name)
        _RERANK_ENCODER_CACHE[model_name] = encoder
    return encoder


def _local_cross_encoder_rerank(query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
    encoder = _get_rerank_encoder(DEFAULT_RERANK_MODEL)
    contents = [c.content for c in chunks]
    scores = list(encoder.rerank(query, contents))
    ranked = sorted(zip(chunks, scores), key=lambda pair: -float(pair[1]))
    result = []
    for chunk, score in ranked[:top_n]:
        chunk.score = float(score)
        result.append(chunk)
    return result


def _jina_rerank(query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is required when RERANK_BACKEND=jina")

    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "top_n": min(top_n, len(chunks)),
            "return_documents": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        JINA_API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            data = body.get("results") or body.get("data") or []
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Jina rerank response: {body}")

            ranked: list[tuple[RetrievedChunk, float]] = []
            for item in data:
                index = item.get("index")
                score = item.get("relevance_score", item.get("score"))
                if index is None or score is None:
                    continue
                index = int(index)
                if index < 0 or index >= len(chunks):
                    continue
                ranked.append((chunks[index], float(score)))

            if not ranked:
                raise RuntimeError(f"Jina rerank returned no usable results: {body}")

            result = []
            for chunk, score in ranked[:top_n]:
                chunk.score = score
                result.append(chunk)
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Jina rerank HTTP {exc.code}: {detail[:500]}")
        except Exception as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    assert last_error is not None
    raise last_error


def _rerank(query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
    if RERANK_BACKEND == "jina":
        return _jina_rerank(query, chunks, top_n=top_n)
    if RERANK_BACKEND == "local":
        return _local_cross_encoder_rerank(query, chunks, top_n=top_n)
    if RERANK_BACKEND in {"none", "disabled", "off"}:
        return chunks[:top_n]
    raise RuntimeError(f"Unsupported RERANK_BACKEND={RERANK_BACKEND!r}")


def _build_filters(document_ids: list[str], tenant_schema: str, platform_tenant_id: str) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"term": {"tenant_schema": f"{tenant_schema}_{platform_tenant_id}"}},
        {"term": {"platform_tenant_id": platform_tenant_id}},
    ]
    if document_ids:
        filters.append({"terms": {"document_id": document_ids}})
    return filters


def _lexical_query_body(query: str, document_ids: list[str], tenant_schema: str, platform_tenant_id: str, size: int) -> dict[str, Any]:
    should: list[dict[str, Any]] = [
        {"match": {"title": {"query": query, "boost": 3.0}}},
        {"match": {"heading": {"query": query, "boost": 3.0}}},
        {"match_phrase": {"passage_text": {"query": query, "boost": 2.0}}},
        {"match": {"passage_text": {"query": query, "operator": "or", "boost": 1.0}}},
        {"match": {"section": {"query": query, "boost": 1.5}}},
    ]
    body: dict[str, Any] = {
        "size": size,
        "track_total_hits": True,
        "_source": {"excludes": ["passage_embedding"]},
        "query": {
            "bool": {
                "filter": _build_filters(document_ids, tenant_schema, platform_tenant_id),
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "highlight": {"fields": {"passage_text": {"fragment_size": 180, "number_of_fragments": 2}}},
    }
    return body


def _neural_query_body(query: str, document_ids: list[str], tenant_schema: str, platform_tenant_id: str, size: int) -> dict[str, Any]:
    return {
        "size": size,
        "track_total_hits": True,
        "_source": {"excludes": ["passage_embedding"]},
        "query": {
            "bool": {
                "filter": _build_filters(document_ids, tenant_schema, platform_tenant_id),
                "must": [
                    {
                        "neural_sparse": {
                            "passage_embedding": {
                                "query_text": query,
                                "model_id": _SPARSE_TOKENIZER_MODEL_ID or OPENSEARCH_SPARSE_TOKENIZER,
                            }
                        }
                    }
                ],
            }
        },
    }


def _search_once(client: Any, query: str, *, tenant_schema: str, platform_tenant_id: str, document_ids: list[str], size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lexical = client.search(index=opensearch_index_name(tenant_schema, platform_tenant_id), body=_lexical_query_body(query, document_ids, tenant_schema, platform_tenant_id, size))
    lexical_hits = lexical.get("hits", {}).get("hits", [])
    neural_hits: list[dict[str, Any]] = []
    if OPENSEARCH_NEURAL_SPARSE:
        try:
            neural = client.search(index=opensearch_index_name(tenant_schema, platform_tenant_id), body=_neural_query_body(query, document_ids, tenant_schema, platform_tenant_id, size))
            neural_hits = neural.get("hits", {}).get("hits", [])
        except Exception:
            neural_hits = []
    return lexical_hits, neural_hits


def _rrf_merge(hit_lists: list[list[dict[str, Any]]], *, top_k: int) -> list[dict[str, Any]]:
    fused: dict[str, float] = {}
    exemplar: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = str(hit.get("_id"))
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            exemplar.setdefault(key, hit)
    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    results = []
    for key, rrf_score in ordered:
        hit = dict(exemplar[key])
        hit["_score"] = rrf_score
        hit["_rrf_score"] = rrf_score
        results.append(hit)
    return results


def _results_to_chunks(hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for idx, hit in enumerate(hits, start=1):
        chunk = _source_to_chunk(hit)
        chunk.evidence_id = f"E{idx}"
        chunks.append(chunk)
    return chunks


async def retrieve_and_rerank(
    query: str,
    *,
    tenant_schema: str,
    platform_tenant_id: str,
    document_ids: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    client = build_opensearch_client()
    index_name = opensearch_index_name(tenant_schema, platform_tenant_id)
    await _to_thread_with_retry(ensure_opensearch_index, client, index_name=index_name)

    def _search():
        lexical, neural = _search_once(
            client,
            query,
            tenant_schema=tenant_schema,
            platform_tenant_id=platform_tenant_id,
            document_ids=document_ids,
            size=max(LEXICAL_CANDIDATE_K, NEURAL_CANDIDATE_K),
        )
        return _rrf_merge([lexical, neural], top_k=max(top_k, RERANK_CANDIDATE_K) if neural else top_k)

    hits = await _to_thread_with_retry(_search)
    chunks = _results_to_chunks(hits)
    if RERANK_ENABLED and chunks:
        candidates = chunks[: max(RERANK_CANDIDATE_K, top_k)]
        try:
            chunks = await _to_thread_with_retry(_rerank, query, candidates, top_n=top_k)
        except Exception:
            # Reranking is an enhancement, not a hard dependency for retrieval.
            # If the hosted service is unavailable, return the fused OpenSearch
            # candidates rather than taking down the user's request.
            chunks = candidates[:top_k]
    return chunks[:top_k]


async def _plan_agentic_queries(model: Any, query: str, *, tried: list[str]) -> list[str]:
    if not model:
        return []
    already = "\n".join(f"- {q}" for q in tried)
    prompt = f"""You are helping an enterprise document search engine recover relevant passages without vector search.\nOriginal question: {query}\nQueries already tried:\n{already}\n\nGenerate up to 3 alternative search queries that preserve the user's intent but use likely document vocabulary, synonyms, exact phrases, abbreviations, or domain terms. Do not invent facts. Avoid repeating the tried queries. Return JSON only: {{\"queries\":[\"...\"]}}"""
    try:
        response = await model.ainvoke(prompt)
        text = content_to_text(response.content).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        payload = json.loads(match.group(0))
        queries = payload.get("queries") or []
        cleaned = []
        seen = {q.strip().lower() for q in tried}
        for q in queries:
            if not isinstance(q, str):
                continue
            q = re.sub(r"\s+", " ", q).strip()
            if q and q.lower() not in seen:
                cleaned.append(q)
                seen.add(q.lower())
        return cleaned[:3]
    except Exception:
        return []


async def retrieve_with_self_correction(
    query: str,
    *,
    model,
    tenant_schema: str,
    platform_tenant_id: str,
    document_ids: list[str],
    top_k: int = DEFAULT_TOP_K,
    max_rounds: int = MAX_AGENTIC_RETRIEVAL_ROUNDS,
) -> tuple[list[RetrievedChunk], str]:
    """Sparse-first retrieval with bounded agentic query expansion.

    First attempt is cheap and deterministic. Only a weak first pass triggers
    an LLM planner, which proposes a few alternate lexical queries. Results
    from all attempts are fused and then optionally reranked. The hard round
    limit prevents runaway tool/LLM usage.
    """
    tried = [query]
    best_chunks = await retrieve_and_rerank(
        query,
        tenant_schema=tenant_schema,
        platform_tenant_id=platform_tenant_id,
        document_ids=document_ids,
        top_k=top_k,
    )
    best_query = query

    if not AGENTIC_RETRIEVAL_ENABLED or len(best_chunks) >= AGENTIC_RETRIEVAL_MIN_HITS:
        return best_chunks, best_query

    for _ in range(max(0, max_rounds)):
        candidates = await _plan_agentic_queries(model, query, tried=tried)
        if not candidates:
            break
        for candidate in candidates:
            tried.append(candidate)
        searches = await asyncio.gather(
            *[
                retrieve_and_rerank(
                    candidate,
                    tenant_schema=tenant_schema,
                    platform_tenant_id=platform_tenant_id,
                    document_ids=document_ids,
                    top_k=top_k,
                )
                for candidate in candidates
            ],
            return_exceptions=True,
        )
        candidate_chunks = [x for x in searches if isinstance(x, list)]
        if not candidate_chunks:
            break
        merged = {c.content: c for c in best_chunks}
        for group in candidate_chunks:
            for chunk in group:
                if chunk.content not in merged or chunk.score > merged[chunk.content].score:
                    merged[chunk.content] = chunk
        merged_list = sorted(merged.values(), key=lambda c: -c.score)[:top_k]
        if merged_list and (not best_chunks or merged_list[0].score > best_chunks[0].score or len(merged_list) > len(best_chunks)):
            best_chunks = merged_list
            best_query = "; ".join(candidates)
        if len(best_chunks) >= AGENTIC_RETRIEVAL_MIN_HITS:
            break
    for i, chunk in enumerate(best_chunks, start=1):
        chunk.evidence_id = f"E{i}"
    return best_chunks, best_query


async def fetch_all_document_chunks(
    document_id: str,
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
    max_chunks: int = DEFAULT_MAX_SUMMARY_CHUNKS,
) -> list[RetrievedChunk]:
    client = build_opensearch_client()
    index_name = opensearch_index_name(tenant_schema, platform_tenant_id)
    await _to_thread_with_retry(ensure_opensearch_index, client, index_name=index_name)

    body = {
        "size": max_chunks,
        "sort": [{"chunk_number": {"order": "asc"}}],
        "query": {
            "bool": {
                "filter": _build_filters([document_id], tenant_schema, platform_tenant_id)
            }
        },
        "_source": {"excludes": ["passage_embedding"]},
    }

    def _search() -> Any:
        return client.search(index=index_name, body=body)

    response = await _to_thread_with_retry(_search)
    chunks = _results_to_chunks(response.get("hits", {}).get("hits", []))
    return chunks[:max_chunks]


async def delete_document(
    document_id: str,
    *,
    tenant_schema: str = TENANT_SCHEMA,
    platform_tenant_id: str = PLATFORM_TENANT_ID,
) -> int:
    client = build_opensearch_client()
    index_name = opensearch_index_name(tenant_schema, platform_tenant_id)
    await _to_thread_with_retry(ensure_opensearch_index, client, index_name=index_name)

    body = {"query": {"bool": {"filter": _build_filters([document_id], tenant_schema, platform_tenant_id)}}}

    def _delete() -> Any:
        response = client.delete_by_query(index=index_name, body=body, refresh=True, conflicts="proceed")
        return response

    await _to_thread_with_retry(_delete)
    count = await _to_thread_with_retry(lambda: client.count(index=index_name, body={"query": body["query"]}))
    return int(count.get("count", 0))


def get_web_search_tool(*, max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS):
    from tools.web_search_tool import WebSearchTool

    config = load_runtime_config()
    api_key = (config.web_search.web_search_api_key or "").strip()
    if not api_key:
        raise RuntimeError("Web search is not configured. Add TAVILY_API_KEY or WEB_SEARCH_API_KEY to .env.")
    return WebSearchTool(api_key=api_key, max_results=max_results or config.web_search.web_search_max_results)


def make_token_tracker():
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    return UsageMetadataCallbackHandler()


def summarize_token_usage(tracker) -> dict:
    input_tokens = sum(v.get("input_tokens", 0) for v in tracker.usage_metadata.values())
    output_tokens = sum(v.get("output_tokens", 0) for v in tracker.usage_metadata.values())
    total_tokens = sum(v.get("total_tokens", 0) for v in tracker.usage_metadata.values())
    return {"total_tokens": total_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens, "by_model": dict(tracker.usage_metadata)}


def merge_web_search_query(*, user_message: str, tool_query: str) -> str:
    from shared.query_merge import merge_web_search_query as _merge

    return _merge(user_message=user_message, tool_query=tool_query)


from citations.verifier import finalize as verifier_finalize  # noqa: E402


def build_langchain_model(backend: str, *, gateway_token: str = ""):
    if backend == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=HF_CHAT_MODEL,
            task="text-generation",
            huggingfacehub_api_token=get_hf_token(),
        )
        return ChatHuggingFace(llm=endpoint)

    if backend == "gemini-api":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_API_MODEL, google_api_key=get_gemini_api_key())

    if backend in {"openrouter", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        api_key = _get_env("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the openrouter backend.")
        return ChatOpenAI(
            model=_get_env("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=_get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            temperature=0,
            timeout=120.0,
            max_retries=5,
        )

    if backend == "litellm":
        from langchain_openai import ChatOpenAI
        from dotenv import dotenv_values

        values = dotenv_values(_POC_DIR / ".env")
        openrouter_key = (values.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
        gemini_key = (values.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        gateway_token = (gateway_token or _get_env("LLM_GATEWAY_API_KEY", "")).strip()
        configured_model = _get_env("LLM_MODEL", "").strip() or _get_env("LLM_GATEWAY_MODEL", "").strip()
        gateway_url = _get_env("LLM_BASE_URL", "").strip() or _get_env("LLM_GATEWAY_URL", "").strip()
        if gateway_token and gateway_url:
            model = "session-model"
        elif configured_model:
            model = configured_model
        elif openrouter_key:
            model = "portfolio-free"
        elif gemini_key:
            model = "portfolio-gemini"
        else:
            raise RuntimeError("No LiteLLM upstream provider credential is configured.")
        return ChatOpenAI(
            model=model,
            base_url=gateway_url or "http://litellm:4000/v1",
            api_key=gateway_token or _get_env("LLM_GATEWAY_API_KEY", ""),
            temperature=0,
            timeout=120.0,
            max_retries=5,
        )

    if backend == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_get_env("VLLM_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=_get_env("VLLM_BASE_URL", "http://vllm:8000/v1"),
            api_key=_get_env("VLLM_API_KEY", "local"),
            temperature=0,
        )

    raise ValueError(f"Unknown backend {backend!r}")


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def build_evidence_registry(chunks: list[RetrievedChunk]) -> dict:
    return {
        "eid_to_doc": {c.evidence_id: c.document_id for c in chunks},
        "eid_to_content": {c.evidence_id: c.content for c in chunks},
    }
