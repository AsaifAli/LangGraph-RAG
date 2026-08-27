"""Request/query entities the retrieval layer
operates on. `SecureAgentRequest.__post_init__` enforces a real
`isinstance(self.query, SanitizedQuery)` check — the query has to actually
be a `SanitizedQuery` instance, not a plain string or dict, so a caller
can't accidentally pass unvalidated input through to retrieval. This app
has no upstream sanitization step of its own (queries come straight from
the Streamlit chat box), so `SanitizedQuery` here validates only
structural non-emptiness rather than redacting anything — the type still
exists to keep the retrieval-layer contract explicit."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class SanitizedInput:
    """Immutable contract guaranteeing that input has been sanitized
    upstream. This POC has no Delivery-layer sanitization step (queries
    come straight from the Streamlit chat box) — `SanitizedQuery(content=q)`
    is constructed directly wherever `SecureAgentRequest` requires one,
    so this validates only structural non-emptiness, never redacts
    anything."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
    sanitization_timestamp: Optional[str] = None
    sanitization_version: Optional[str] = None

    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("Sanitized content cannot be empty")


@dataclass(frozen=True)
class SanitizedQuery(SanitizedInput):
    pass


@dataclass(frozen=True)
class SanitizedConversationContext:
    history: list
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.history is None:
            object.__setattr__(self, "history", [])


@dataclass(frozen=True)
class AccessibleDocumentMeta:
    """Referenced only as a SecureAgentRequest field type — this POC never
    actually constructs one (no chat-service document-management caller)."""

    document_id: UUID
    document_name: str
    document_type: str
    groups: List[str]
    tags: List[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None


@dataclass(frozen=True)
class SecureAgentRequest:
    """Immutable secure agent request with enforced sanitization.

    the retrieval layer reads
    `.tenant_schema`, `.platform_tenant_id`, `.document_ids`, `.person_ids`,
    `.org_ids` off this; rag_pipeline.py constructs one per retrieval call.
    """

    query: SanitizedQuery
    correlation_id: str
    platform_tenant_id: str
    conversation_context: Optional[SanitizedConversationContext] = None

    document_ids: Optional[List[UUID]] = None
    document_ids_are_explicit_scope: bool = True

    person_ids: Optional[List[str]] = None
    org_ids: Optional[List[str]] = None

    user_id: Optional[int] = None
    session_id: Optional[str] = None
    tenant_schema: Optional[str] = None

    metadata: Optional[Dict[str, Any]] = None

    process_conversation_title: bool = False
    conversation_title_query: Optional[str] = None

    is_temp: bool = False
    auth_token: Optional[str] = None
    accessible_document_metadata: Optional[List[AccessibleDocumentMeta]] = None

    summary_scope_type: Optional[str] = None
    summary_scope_value: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.query, SanitizedQuery):
            raise TypeError(
                "query must be SanitizedQuery. Sanitization MUST be performed in Delivery layer."
            )

        if self.conversation_context is not None and not isinstance(
            self.conversation_context, SanitizedConversationContext
        ):
            raise TypeError("conversation_context must be SanitizedConversationContext.")

        if not self.correlation_id:
            raise ValueError("correlation_id is required for observability and tracing.")

        if self.document_ids is not None:
            if not isinstance(self.document_ids, list):
                raise TypeError("document_ids must be a list of UUID.")
            for doc_id in self.document_ids:
                if not isinstance(doc_id, UUID):
                    raise TypeError("Each document_id must be UUID.")
