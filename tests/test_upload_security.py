import io
import zipfile

import pytest
from PIL import Image

from shared.upload_security import validate_document_upload


def _docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    return buffer.getvalue()


def test_accepts_supported_text_and_pdf_signatures():
    validate_document_upload(b"Supplier: Example\nPrice: 42", "quote.txt")
    validate_document_upload(b"%PDF-1.7\nbody", "quote.pdf")
    validate_document_upload(b"Heading\nValue", "notes.md")
    validate_document_upload(b"name,cost\nA,10", "quotes.csv")


def test_rejects_pdf_extension_with_non_pdf_content():
    with pytest.raises(ValueError, match="PDF"):
        validate_document_upload(b"not a pdf", "fake.pdf")


def test_accepts_valid_docx_and_rejects_malformed_office_archive():
    validate_document_upload(_docx_bytes(), "quote.docx")
    malformed = io.BytesIO()
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("not-the-document.xml", "<doc/>")

    with pytest.raises(ValueError, match="DOCX"):
        validate_document_upload(malformed.getvalue(), "fake.docx")


def test_accepts_legacy_doc_signature():
    ole_header = bytes.fromhex("D0CF11E0A1B11AE1") + b"legacy-doc"
    validate_document_upload(ole_header, "legacy.doc")


def test_validates_image_bytes_and_extension_match():
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image_buffer, format="PNG")
    png_bytes = image_buffer.getvalue()

    validate_document_upload(png_bytes, "diagram.png")
    with pytest.raises(ValueError, match="image"):
        validate_document_upload(png_bytes, "diagram.jpg")


def test_rejects_empty_binary_lookalike_text_and_unsupported_types():
    with pytest.raises(ValueError, match="empty"):
        validate_document_upload(b"", "empty.txt")
    with pytest.raises(ValueError, match="text"):
        validate_document_upload(b"hello\x00world", "fake.txt")
    with pytest.raises(ValueError, match="Unsupported"):
        validate_document_upload(b"anything", "payload.exe")


def test_enforces_configurable_upload_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_DOCUMENT_UPLOAD_MB", "0.000001")
    with pytest.raises(ValueError, match="size limit"):
        validate_document_upload(b"xx", "quote.txt")
