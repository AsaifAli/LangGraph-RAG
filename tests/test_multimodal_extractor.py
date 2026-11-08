from io import BytesIO
from types import SimpleNamespace

import fitz
import pytest
from PIL import Image
from docx import Document

from shared.multimodal_extractor import (
    extract_multimodal_sections,
)


class FakeVisionModel:

    def __init__(
        self,
        text="A chart showing revenue rising from 10 to 20.",
    ):
        self.text = text
        self.calls = 0

    async def ainvoke(self, messages):

        self.calls += 1

        return SimpleNamespace(
            content=self.text
        )


def make_png() -> bytes:

    image = Image.new(
        "RGB",
        (48, 32),
        color="white",
    )

    output = BytesIO()

    image.save(
        output,
        format="PNG",
    )

    return output.getvalue()


def make_pdf(*page_text: str) -> bytes:

    document = fitz.open()

    for text in page_text:

        page = document.new_page()

        if text:
            page.insert_text(
                (72, 72),
                text,
            )

    payload = document.tobytes()

    document.close()

    return payload


def make_docx_with_image() -> bytes:

    document = Document()

    document.add_paragraph(
        "Quarterly business report"
    )

    document.add_picture(
        BytesIO(make_png())
    )

    output = BytesIO()

    document.save(output)

    return output.getvalue()


@pytest.mark.asyncio
async def test_standalone_image_requires_multimodal_flag():

    with pytest.raises(
        ValueError,
        match="MULTIMODAL_INGEST_ENABLED",
    ):

        await extract_multimodal_sections(
            make_png(),
            "chart.png",
            multimodal_enabled=False,
        )


@pytest.mark.asyncio
async def test_standalone_image_becomes_visual_evidence():

    model = FakeVisionModel(
        "The bar chart shows revenue increasing from 10 to 20."
    )

    result = await extract_multimodal_sections(
        make_png(),
        "chart.png",
        vision_model=model,
        multimodal_enabled=True,
    )

    assert len(result.sections) == 1

    section = result.sections[0]

    assert section.modality == "visual"

    assert "revenue increasing" in section.text

    assert result.visual_pages_processed == 1

    assert model.calls == 1


@pytest.mark.asyncio
async def test_pdf_combines_text_and_visual_evidence():

    model = FakeVisionModel(
        "A process diagram connects intake to approval."
    )

    result = await extract_multimodal_sections(
        make_pdf(
            "Policy scope is shown on this page."
        ),
        "policy.pdf",
        vision_model=model,
        multimodal_enabled=True,
    )

    assert len(result.sections) == 1

    section = result.sections[0]

    assert section.page_no == 1

    assert section.modality == "mixed"

    assert "Policy scope" in section.text

    assert "process diagram" in section.text

    assert result.visual_pages_processed == 1


@pytest.mark.asyncio
async def test_pdf_page_limit_preserves_remaining_text():

    model = FakeVisionModel()

    result = await extract_multimodal_sections(
        make_pdf(
            "Page one text.",
            "Page two text.",
        ),
        "two-pages.pdf",
        vision_model=model,
        multimodal_enabled=True,
        max_pdf_pages=1,
    )

    assert [
        section.page_no
        for section in result.sections
    ] == [1, 2]

    assert result.visual_pages_processed == 1

    assert result.visual_pages_skipped == 1

    assert "Page two text" in result.sections[1].text

    assert model.calls == 1


@pytest.mark.asyncio
async def test_text_pdf_works_without_multimodal_mode():

    result = await extract_multimodal_sections(
        make_pdf(
            "The deductible is CAD 500."
        ),
        "policy.pdf",
        multimodal_enabled=False,
    )

    section = result.sections[0]

    assert section.page_no == 1

    assert section.modality == "text"

    assert "CAD 500" in section.text

    assert result.visual_pages_processed == 0


@pytest.mark.asyncio
async def test_scanned_pdf_without_vision_fails_clearly():

    with pytest.raises(
        ValueError,
        match="Enable multimodal ingestion",
    ):

        await extract_multimodal_sections(
            make_pdf(""),
            "scan.pdf",
            multimodal_enabled=False,
        )


@pytest.mark.asyncio
async def test_docx_embedded_image_is_analyzed():

    model = FakeVisionModel(
        "The embedded image contains an architecture diagram."
    )

    result = await extract_multimodal_sections(
        make_docx_with_image(),
        "architecture.docx",
        vision_model=model,
        multimodal_enabled=True,
    )

    assert any(
        section.modality == "visual"
        for section in result.sections
    )

    assert any(
        "architecture diagram" in section.text
        for section in result.sections
    )

    assert model.calls == 1