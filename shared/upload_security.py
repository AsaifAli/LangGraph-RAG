"""Server-side upload validation for EvidenceFlow's shared ingestion path."""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

_TEXT_SUFFIXES = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".log", ".html", ".htm", ".eml", ".sql", ".py",
    ".js", ".ts", ".css", ".sh", ".conf",
}
_OFFICE_MEMBERS = {
    ".docx": "word/document.xml",
    ".xlsx": "xl/workbook.xml",
    ".xlsm": "xl/workbook.xml",
    ".xlsb": "xl/workbook.bin",
    ".pptx": "ppt/presentation.xml",
}
_ODF_MIMETYPES = {
    ".odt": b"application/vnd.oasis.opendocument.text",
    ".ods": b"application/vnd.oasis.opendocument.spreadsheet",
    ".odp": b"application/vnd.oasis.opendocument.presentation",
}
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
_OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


def _max_upload_bytes() -> int:
    raw_value = os.getenv("MAX_DOCUMENT_UPLOAD_MB", "200").strip()
    try:
        megabytes = float(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_DOCUMENT_UPLOAD_MB must be a positive number.") from exc
    if megabytes <= 0:
        raise ValueError("MAX_DOCUMENT_UPLOAD_MB must be a positive number.")
    return int(megabytes * 1024 * 1024)


def validate_declared_upload_size(size: int | None) -> None:
    """Reject oversized uploads before copying their bytes into app memory."""
    if size is None:
        return
    try:
        declared = int(size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Uploaded document has an invalid declared size.") from exc
    if declared < 0 or declared > _max_upload_bytes():
        raise ValueError("Uploaded document exceeds the configured size limit.")


def _validate_zip_container(content: bytes, *, suffix: str, expected_member: str, max_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if "[Content_Types].xml" not in names or expected_member not in names:
                raise ValueError(f"File contents do not match {suffix[1:].upper()} format.")
            _validate_zip_members(members, max_bytes=max_bytes)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"File contents do not match {suffix[1:].upper()} format.") from exc


def _validate_zip_members(members: list[zipfile.ZipInfo], *, max_bytes: int) -> None:
    if len(members) > _MAX_OFFICE_ENTRIES:
        raise ValueError("Office document contains too many archive entries.")
    expanded_size = 0
    for member in members:
        if member.flag_bits & 0x1:
            raise ValueError("Encrypted Office documents are not supported.")
        if member.filename.startswith(("/", "\\")) or ".." in Path(member.filename).parts:
            raise ValueError("Office document contains an unsafe archive path.")
        expanded_size += member.file_size
        if expanded_size > min(_MAX_OFFICE_EXPANDED_BYTES, max(max_bytes * 20, max_bytes)):
            raise ValueError("Office document expands beyond the safe size limit.")


def _validate_odf(content: bytes, *, suffix: str, max_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if "mimetype" not in {member.filename for member in members} or "content.xml" not in {member.filename for member in members}:
                raise ValueError(f"File contents do not match {suffix[1:].upper()} format.")
            _validate_zip_members(members, max_bytes=max_bytes)
            if archive.read("mimetype").strip() != _ODF_MIMETYPES[suffix]:
                raise ValueError(f"File contents do not match {suffix[1:].upper()} format.")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"File contents do not match {suffix[1:].upper()} format.") from exc


def validate_document_upload(content: bytes, source_name: str) -> None:
    """Validate size and file signatures before parsers or model calls run.

    Text: TXT, MD, CSV, TSV, JSON, XML, YAML, TOML, INI, LOG, HTML, EML,
    SQL and common source/config files.
    Office: DOC, DOCX, XLS, XLSX, XLSM, XLSB, ODT, ODS, PPTX and ODP.
    Also PDF and the supported raster image formats.
    """
    if not isinstance(content, bytes) or not content:
        raise ValueError("Uploaded document is empty.")

    max_bytes = _max_upload_bytes()
    if len(content) > max_bytes:
        raise ValueError(f"Uploaded document exceeds the {max_bytes}-byte size limit.")

    suffix = Path(source_name or "").suffix.lower()

    if suffix == ".pdf":
        if b"%PDF-" not in content[:1024]:
            raise ValueError("File contents do not match PDF format.")
        return

    if suffix in _OFFICE_MEMBERS:
        _validate_zip_container(
            content,
            suffix=suffix,
            expected_member=_OFFICE_MEMBERS[suffix],
            max_bytes=max_bytes,
        )
        return

    if suffix in _ODF_MIMETYPES:
        _validate_odf(content, suffix=suffix, max_bytes=max_bytes)
        return

    if suffix in {".doc", ".xls", ".ppt"}:
        if not content.startswith(_OLE_SIGNATURE):
            label = {".doc": "legacy DOC", ".xls": "legacy XLS", ".ppt": "legacy PPT"}[suffix]
            raise ValueError(f"File contents do not match {label} format.")
        return

    if suffix == ".rtf":
        if not content.lstrip().startswith(b"{\\rtf"):
            raise ValueError("File contents do not match RTF format.")
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
        "Supported: PDF, DOC/DOCX, XLS/XLSX/XLSM/XLSB, PPT/PPTX/ODP, ODT/ODS, "
        "RTF, TXT, MD, CSV, TSV, JSON, XML, YAML, TOML, INI, LOG, HTML, "
        "EML, SQL, source/config text and PNG/JPG/WEBP/BMP/TIFF."
    )
