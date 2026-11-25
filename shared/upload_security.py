"""Server-side upload validation for EvidenceFlow's shared ingestion path."""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md", ".csv"}
_IMAGE_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
_MAX_OFFICE_ENTRIES = 10_000
_MAX_OFFICE_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000


def _max_upload_bytes() -> int:
    raw_value = os.getenv("MAX_DOCUMENT_UPLOAD_MB", "50").strip()
    try:
        megabytes = float(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_DOCUMENT_UPLOAD_MB must be a positive number.") from exc
    if megabytes <= 0:
        raise ValueError("MAX_DOCUMENT_UPLOAD_MB must be a positive number.")
    return int(megabytes * 1024 * 1024)


def _validate_office_archive(content: bytes, *, suffix: str, max_bytes: int) -> None:
    expected_member = "word/document.xml" if suffix == ".docx" else "xl/workbook.xml"
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if (
                "[Content_Types].xml" not in names
                or expected_member not in names
            ):
                raise ValueError(
                    f"File contents do not match {suffix[1:].upper()} format."
                )
            if len(members) > _MAX_OFFICE_ENTRIES:
                raise ValueError("Office document contains too many archive entries.")
            expanded_size = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("Encrypted Office documents are not supported.")
                if member.filename.startswith(("/", "\\")) or ".." in Path(member.filename).parts:
                    raise ValueError("Office document contains an unsafe archive path.")
                expanded_size += member.file_size
                if expanded_size > min(
                    _MAX_OFFICE_EXPANDED_BYTES,
                    max(max_bytes * 20, max_bytes),
                ):
                    raise ValueError("Office document expands beyond the safe size limit.")
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"File contents do not match {suffix[1:].upper()} format."
        ) from exc


def validate_document_upload(content: bytes, source_name: str) -> None:
    """Reject oversized or mislabeled uploads before parsers and model calls run.

    Supported formats are intentionally aligned with the existing ingestion
    pipeline: PDF, DOC, DOCX, TXT, MD, CSV and the existing image suffixes.
    This function validates the upload; it does not change extraction behavior.
    """
    if not isinstance(content, bytes) or not content:
        raise ValueError("Uploaded document is empty.")

    max_bytes = _max_upload_bytes()
    if len(content) > max_bytes:
        raise ValueError(
            f"Uploaded document exceeds the {max_bytes}-byte size limit."
        )

    suffix = Path(source_name or "").suffix.lower()

    if suffix == ".pdf":
        if b"%PDF-" not in content[:1024]:
            raise ValueError("File contents do not match PDF format.")
        return

    if suffix == ".docx":
        _validate_office_archive(content, suffix=suffix, max_bytes=max_bytes)
        return

    if suffix == ".doc":
        if not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
            raise ValueError("File contents do not match legacy DOC format.")
        return

    if suffix in _IMAGE_FORMATS:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format != _IMAGE_FORMATS[suffix]:
                    raise ValueError("Uploaded image bytes do not match the file extension.")
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                    raise ValueError("Uploaded image dimensions exceed the safe limit.")
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded image is invalid or corrupted.") from exc
        return

    if suffix in _TEXT_SUFFIXES:
        if b"\x00" in content[:8192]:
            raise ValueError("File contents do not appear to be a supported text document.")
        return

    raise ValueError(
        f"Unsupported upload type: {suffix or '(no extension)'}. "
        "Supported: PDF, DOC, DOCX, TXT, MD, CSV, PNG, JPG, WEBP, BMP and TIFF."
    )
