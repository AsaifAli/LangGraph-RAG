# Portfolio Positioning

## One-line pitch

**Evidence-Grounded Agentic RAG Platform** — a LangGraph research system that combines hybrid retrieval, cross-encoder reranking, web research, persistent memory, and fail-closed citation QA.

## What to demonstrate

1. **Grounded answer:** ask for the 2025 Cyber Liability deductible and open the citation evidence.
2. **Cross-document comparison:** compare the 2025 and 2026 policies and show changed limits/deductibles.
3. **Abstention:** ask for a fact that is absent from the documents and show the system refusing to invent it.
4. **Quality inspection:** show citation coverage, numeric/date support, and possible evidence conflicts.
5. **Evaluation:** run the benchmark and commit the generated report when you have a real model-backed run.

## Why it stands out

The project is not presented as a generic chatbot. Its core engineering problem is **evidence quality**: the system closes the evidence set for each turn, verifies emitted citations against that set, checks material numeric/date claims deterministically, and fails closed when a KB route produces no usable evidence.

## Recruiter talking points

- Why hybrid retrieval instead of dense-only search?
- Why rerank after RRF?
- How does LangGraph manage conditional routing and fan-out?
- How are citations authorized and verified?
- What happens when evidence is missing or conflicting?
- How would you evaluate retrieval independently from generation?
- What would you change for asynchronous ingestion and multi-tenant production deployment?
