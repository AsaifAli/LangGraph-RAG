
"""Guided first-run experience for the public EvidenceFlow demonstration."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

SAMPLE_DOCUMENTS = (
    {
        "title": "Northbridge policy · 2025",
        "filename": "policy_2025.md",
        "description": "Baseline policy covering cyber, business interruption, and data-breach terms.",
    },
    {
        "title": "Northbridge policy · 2026",
        "filename": "policy_2026.md",
        "description": "Updated policy values for cross-document comparison.",
    },
)

EXAMPLE_QUESTIONS = (
    (
        "01 · Grounded lookup",
        "What is the Cyber Liability sublimit and deductible?",
    ),
    (
        "02 · Cross-document comparison",
        "Compare the Cyber Liability and Business Interruption changes between the 2025 and 2026 policies.",
    ),
    (
        "03 · Evidence abstention",
        "What is the company's revenue in 2030?",
    ),
)


def render_demo_onboarding(demo_dir: Path) -> None:
    """Show synthetic source files and research prompts on an empty workspace."""
    available = [
        (sample, demo_dir / sample["filename"])
        for sample in SAMPLE_DOCUMENTS
        if (demo_dir / sample["filename"]).is_file()
    ]

    st.markdown("#### Start with the sample policy documents")
    st.caption(
        "Download both synthetic files, then upload them under Documents "
        "(all chats) in the sidebar or attach them to this chat. These files "
        "demonstrate retrieval, comparison, citations, and abstention."
    )

    if available:
        columns = st.columns(len(available), gap="medium")

        for index, (sample, path) in enumerate(available):
            with columns[index]:
                with st.container(border=True):
                    st.markdown(f"**{sample['title']}**")
                    st.caption(sample["description"])
                    st.download_button(
                        "Download sample file",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/markdown",
                        key=f"evidenceflow-sample-{path.stem}",
                        use_container_width=True,
                    )
    else:
        st.info("Sample policy documents are not included in this deployment.")

    st.markdown("#### Try these research questions")
    st.caption(
        "Copy a question from a card below. Upload both policies before "
        "trying the cross-document comparison."
    )

    columns = st.columns(3, gap="medium")

    for index, (title, question) in enumerate(EXAMPLE_QUESTIONS):
        with columns[index]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.code(question, language=None)
