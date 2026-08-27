"""Shared constants used across the retrieval/citation/ingestion pipeline:
retrieval metadata keys, the rerank model id, and the error-code/
message strings `shared/exceptions.py` and `shared/text_extractors.py`
raise. Kept to exactly the fields this codebase actually reads — no
unused classes carried along "just in case"."""

import os


class AIConstants:
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "jina-reranker-v1-turbo-en")
    COLLECTION_SUFFIX: str = "embeddings"


class RetrievalConstants:
    META_DOCUMENT_ID_KEY = "document_id"
    META_THREAD_ID_KEY = "thread_id"
    META_TENANT_SCHEMA_KEY = "tenant_schema"
    META_PERSON_IDS_KEY = "person_ids"
    META_ORG_IDS_KEY = "org_ids"


class AIReferenceKeys:
    """Keys used in reference extraction and response metadata."""

    META_DATA = "meta_data"
    CONTENT = "content"


class ErrorMessages:
    APPLICATION_ERROR = "APPLICATION_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"

    LIBREOFFICE_NOT_FOUND = "LibreOffice not found on system"
    LIBREOFFICE_CONVERSION_FAILED = "LibreOffice conversion failed"
    LIBREOFFICE_NO_OUTPUT_FILE = "LibreOffice did not produce output text file"

    PDF_TEXT_EXTRACTION_FAILED = "Failed to extract text from PDF"
    PDF_PARSED_BUT_EMPTY = "PDF parsed but no text extracted"

    DOCX_INVALID_OR_CORRUPTED = "Invalid or corrupted DOCX file"
    DOCX_PARSED_BUT_EMPTY = "DOCX parsed but no text extracted"

    DOC_CONVERTED_TEXT_EMPTY = "DOC converted but output text is empty"
    TXT_FILE_EMPTY = "TXT file is empty."
    TXT_DECODING_FAILED = "Failed to decode TXT file content."
