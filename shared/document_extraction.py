"""Bounded text extraction for common business and office document formats."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

_MAX_SHEETS = 200
_MAX_ROWS_PER_SHEET = 20_000
_MAX_COLUMNS_PER_ROW = 256
_MAX_SLIDES = 500


class _VisibleHtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth and data.strip():
            self.parts.append(data.strip())


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _extract_xlsx(content: bytes) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(
        io.BytesIO(content),
        read_only=True,
        data_only=True,
        keep_links=False,
    )
    sections: list[str] = []
    try:
        for sheet_index, sheet in enumerate(workbook.worksheets):
            if sheet_index >= _MAX_SHEETS:
                sections.append("[Additional worksheets omitted: sheet limit reached.]")
                break
            rows: list[str] = [f"Worksheet: {sheet.title}"]
            for row_index, values in enumerate(
                sheet.iter_rows(values_only=True),
                start=1,
            ):
                if row_index > _MAX_ROWS_PER_SHEET:
                    rows.append("[Additional rows omitted: worksheet row limit reached.]")
                    break
                cells = [_format_cell(value) for value in values[:_MAX_COLUMNS_PER_ROW]]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    rows.append(" | ".join(cells))
            if len(rows) > 1:
                sections.append("\n".join(rows))
    finally:
        workbook.close()
    return "\n\n".join(sections)


def _extract_xls(content: bytes) -> str:
    import xlrd

    workbook = xlrd.open_workbook(file_contents=content, on_demand=True)
    sections: list[str] = []
    try:
        for sheet_index in range(min(workbook.nsheets, _MAX_SHEETS)):
            sheet = workbook.sheet_by_index(sheet_index)
            rows = [f"Worksheet: {sheet.name}"]
            for row_index in range(min(sheet.nrows, _MAX_ROWS_PER_SHEET)):
                cells = [
                    _format_cell(sheet.cell_value(row_index, col_index))
                    for col_index in range(min(sheet.ncols, _MAX_COLUMNS_PER_ROW))
                ]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    rows.append(" | ".join(cells))
            if len(rows) > 1:
                sections.append("\n".join(rows))
    finally:
        workbook.release_resources()
    return "\n\n".join(sections)


def _extract_xlsb(content: bytes) -> str:
    from pyxlsb import open_workbook

    sections: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".xlsb") as temporary:
        temporary.write(content)
        temporary.flush()
        with open_workbook(temporary.name) as workbook:
            for sheet_index, sheet_name in enumerate(workbook.sheets):
                if sheet_index >= _MAX_SHEETS:
                    sections.append("[Additional worksheets omitted: sheet limit reached.]")
                    break
                rows = [f"Worksheet: {sheet_name}"]
                with workbook.get_sheet(sheet_name) as sheet:
                    for row_index, row in enumerate(sheet.rows(sparse=True), start=1):
                        if row_index > _MAX_ROWS_PER_SHEET:
                            rows.append("[Additional rows omitted: worksheet row limit reached.]")
                            break
                        cells = [_format_cell(cell.v) for cell in row[:_MAX_COLUMNS_PER_ROW]]
                        while cells and not cells[-1]:
                            cells.pop()
                        if any(cells):
                            rows.append(" | ".join(cells))
                if len(rows) > 1:
                    sections.append("\n".join(rows))
    return "\n\n".join(sections)


def _extract_ods(content: bytes) -> str:
    from odf.opendocument import load
    from odf.table import Table, TableCell, TableRow
    from odf.teletype import extractText

    document = load(io.BytesIO(content))
    sections: list[str] = []
    for sheet_index, table in enumerate(document.spreadsheet.getElementsByType(Table)):
        if sheet_index >= _MAX_SHEETS:
            sections.append("[Additional worksheets omitted: sheet limit reached.]")
            break
        title = table.getAttribute("name") or f"Sheet {sheet_index + 1}"
        rows = [f"Worksheet: {title}"]
        for row_index, row in enumerate(table.getElementsByType(TableRow), start=1):
            if row_index > _MAX_ROWS_PER_SHEET:
                rows.append("[Additional rows omitted: worksheet row limit reached.]")
                break
            values = []
            for cell in row.getElementsByType(TableCell)[:_MAX_COLUMNS_PER_ROW]:
                values.append(_format_cell(extractText(cell)))
            while values and not values[-1]:
                values.pop()
            if any(values):
                rows.append(" | ".join(values))
        if len(rows) > 1:
            sections.append("\n".join(rows))
    return "\n\n".join(sections)


def _extract_pptx(content: bytes) -> str:
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(content))
    sections: list[str] = []
    for slide_index, slide in enumerate(presentation.slides):
        if slide_index >= _MAX_SLIDES:
            sections.append("[Additional slides omitted: slide limit reached.]")
            break
        parts = [f"Slide {slide_index + 1}"]
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text.strip()
                if text:
                    parts.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    values = [_format_cell(cell.text) for cell in row.cells]
                    if any(values):
                        parts.append(" | ".join(values))
        if len(parts) > 1:
            sections.append("\n".join(parts))
    return "\n\n".join(sections)



def _extract_legacy_ppt(content: bytes) -> str:
    """Convert legacy binary PowerPoint to text with the bounded local office tool."""
    with tempfile.TemporaryDirectory(prefix="evidenceflow-ppt-") as directory:
        source = Path(directory) / "source.ppt"
        source.write_bytes(content)
        try:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    directory,
                    str(source),
                ],
                check=True,
                capture_output=True,
                timeout=45,
            )
        except FileNotFoundError as exc:
            raise ValueError("Legacy PPT extraction requires LibreOffice Impress in the API image.") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Legacy PPT conversion exceeded the processing time limit.") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("The legacy PPT document could not be converted.") from exc
        converted = Path(directory) / "source.txt"
        if not converted.is_file():
            raise ValueError("The legacy PPT document did not contain extractable text.")
        return _decode_text(converted.read_bytes()).strip()

def _extract_odt(content: bytes) -> str:
    from odf.opendocument import load
    from odf.teletype import extractText

    document = load(io.BytesIO(content))
    return extractText(document.text).strip()


def _extract_odp(content: bytes) -> str:
    from odf.opendocument import load
    from odf.presentation import Page
    from odf.teletype import extractText

    document = load(io.BytesIO(content))
    slides = document.getElementsByType(Page)
    sections = []
    for index, slide in enumerate(slides[:_MAX_SLIDES], start=1):
        text = extractText(slide).strip()
        if text:
            sections.append(f"Slide {index}\n{text}")
    return "\n\n".join(sections)


def extract_document_text(content: bytes, source_name: str) -> str:
    """Extract plain text while retaining sheet, slide and table boundaries."""
    suffix = Path(source_name or "").suffix.lower()

    if suffix in {".xlsx", ".xlsm"}:
        return _extract_xlsx(content)
    if suffix == ".xls":
        return _extract_xls(content)
    if suffix == ".xlsb":
        return _extract_xlsb(content)
    if suffix == ".ods":
        return _extract_ods(content)
    if suffix == ".ppt":
        return _extract_legacy_ppt(content)
    if suffix == ".pptx":
        return _extract_pptx(content)
    if suffix == ".odp":
        return _extract_odp(content)
    if suffix == ".odt":
        return _extract_odt(content)
    if suffix == ".rtf":
        from striprtf.striprtf import rtf_to_text
        return rtf_to_text(_decode_text(content)).strip()
    if suffix in {".html", ".htm"}:
        parser = _VisibleHtmlText()
        parser.feed(_decode_text(content))
        return "\n".join(parser.parts)
    if suffix == ".docx":
        from shared.text_extractors import DocumentTextExtractor
        return DocumentTextExtractor().docx_bytes_to_text(content)
    if suffix == ".doc":
        from shared.text_extractors import DocumentTextExtractor
        return DocumentTextExtractor().doc_bytes_to_text(content)
    if suffix == ".pdf":
        from shared.text_extractors import DocumentTextExtractor
        return DocumentTextExtractor().pdf_bytes_to_text(content)

    text = _decode_text(content)
    if suffix == ".json":
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text
    if suffix == ".tsv":
        rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
        return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
    if suffix == ".csv":
        return text
    return text
