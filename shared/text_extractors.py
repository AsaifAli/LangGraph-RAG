"""Text extraction for uploaded documents: PDF (`pdfplumber`), DOCX
(`python-docx`, including table cells joined with ` | ` per row), legacy
binary `.doc` (a LibreOffice-headless subprocess fallback, since `.doc`
has no reliable pure-Python parser), and TXT (UTF-8 with a latin-1
fallback for files that aren't valid UTF-8).
"""

import subprocess
import tempfile
from pathlib import Path
from io import BytesIO

import pdfplumber
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from shared.exceptions import StorageError
from shared.constants import ErrorMessages
from config.loader import get_config


class DocumentTextExtractor:

    def doc_bytes_to_text(self, doc_bytes: bytes) -> str:
        """
        Try DOCX first, fallback to legacy DOC extraction.
        """
        try:
            return self.docx_bytes_to_text(doc_bytes)
        except Exception as e:
            pass

        return self.doc_bytes_to_text_via_libreoffice(doc_bytes)


    def docx_bytes_to_text(self, docx_bytes: bytes) -> str:
        try:
            doc = Document(BytesIO(docx_bytes))
        except Exception as e:
            raise StorageError(ErrorMessages.DOCX_INVALID_OR_CORRUPTED) from e

        lines: list[str] = []

        def iter_block_items(parent):
            for child in parent.element.body.iterchildren():
                if child.tag.endswith("}p"):
                    yield Paragraph(child, parent)
                elif child.tag.endswith("}tbl"):
                    yield Table(child, parent)

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    lines.append(text)

            elif isinstance(block, Table):
                for row in block.rows:
                    row_text = " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    )
                    if row_text:
                        lines.append(row_text)

        if not lines:
            raise StorageError(ErrorMessages.DOCX_PARSED_BUT_EMPTY)

        return "\n".join(lines)


    def pdf_bytes_to_text(self, pdf_bytes: bytes) -> str:
        text_chunks: list[str] = []

        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
        except Exception as e:
            raise StorageError(ErrorMessages.PDF_TEXT_EXTRACTION_FAILED) from e

        if not text_chunks:
            raise StorageError(ErrorMessages.PDF_PARSED_BUT_EMPTY)

        return "\n\n".join(text_chunks)


    def doc_bytes_to_text_via_libreoffice(self, content_bytes: bytes) -> str:
        LIBREOFFICE = get_config().lib.libreoffice_path
        if not LIBREOFFICE:
            raise StorageError(ErrorMessages.LIBREOFFICE_NOT_FOUND)
        with tempfile.TemporaryDirectory() as tmp:
            doc_path = Path(tmp) / "input.doc"
            doc_path.write_bytes(content_bytes)

            try:
                subprocess.run(
                    [
                        LIBREOFFICE,
                        "--headless",
                        "--convert-to",
                        "txt",
                        "--outdir",
                        tmp,
                        str(doc_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except FileNotFoundError as e:
                raise StorageError(ErrorMessages.LIBREOFFICE_NOT_FOUND) from e

            except subprocess.CalledProcessError as e:
                raise StorageError(
                    ErrorMessages.LIBREOFFICE_CONVERSION_FAILED
                ) from e

            txt_path = doc_path.with_suffix(".txt")
            if not txt_path.exists():
                raise StorageError(ErrorMessages.LIBREOFFICE_NO_OUTPUT_FILE)

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                raise StorageError(ErrorMessages.DOC_CONVERTED_TEXT_EMPTY)

            return text

    def txt_bytes_to_text(self, content: bytes) -> str:
        if not content or content.strip() == b"":
            raise StorageError(ErrorMessages.TXT_FILE_EMPTY)

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1", errors="ignore")
            except Exception as e:
                raise StorageError(ErrorMessages.TXT_DECODING_FAILED) from e
