# EvidenceFlow API architecture

## Request flow

    Authenticated client workspace
      |
      v
    Studio same-origin BFF /api/v1
      | workspace identity + scoped HttpOnly app cookie
      v
    EvidenceFlow FastAPI
      | session exchange / revocation
      | chats and conversation metadata
      | document upload / deletion
      | Server-Sent Events for status, safe task lifecycle, tokens and final outcome
      |
      +--> LangGraph supervisor and evidence auditor
      +--> sparse-first OpenSearch retrieval
      +--> optional web / primary-source research
      +--> shared Portfolio Gateway inference
      +--> owner-scoped quotas and API session store
      - SQLite LangGraph checkpointer

## Retrieval strategy

The knowledge base uses OpenSearch lexical retrieval with BM25, exact/phrase/fuzzy matching, field boosting, and tenant/document filtering. When enabled, OpenSearch neural sparse retrieval adds learned sparse semantic signals while continuing to use an inverted-index-based search path. Candidate lists are fused with reciprocal-rank fusion and optionally reranked with a cross-encoder.

For difficult queries, the LangGraph retrieval layer can ask the LLM to generate a bounded set of alternative lexical queries. Those alternatives are executed by OpenSearch.

## Evidence and QA

Retrieved chunks carry stable evidence IDs and source metadata. Final-answer evidence markers are checked against the exact evidence set supplied to the model. Guardrails treat retrieved documents and web results as untrusted data and prevent retrieved text from authorizing tool behavior.

## Document ingestion

The API validates upload size, file signature, Office/ODF package members, archive entry count, archive paths and expanded size before parsing. Spreadsheet extractors preserve worksheet names, rows and values; presentation and OpenDocument extractors preserve slide/sheet structure. Extracted sections are chunked and indexed under the verified owner scope.

The research stream reports approved LangGraph node lifecycle events using static client-safe labels. This makes agent execution observable without exposing private prompts, chain-of-thought, model internals or raw tool output. The independent evidence auditor remains the release gate for answer tokens.

## Persistence

LangGraph's SQLite checkpointer stores conversational state. Workspace metadata and encrypted short-lived Gateway session records use the configured API store. OpenSearch retains document chunks and retrieval metadata. Runtime state is intentionally excluded from source control.
