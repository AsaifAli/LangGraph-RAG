# 90-Second Portfolio Demo

## 0–15s — Introduce

"This is an evidence-grounded agentic RAG platform built with LangGraph. It can answer from uploaded documents, research the web when needed, and verify the evidence behind generated answers."

## 15–35s — Grounded answer

Upload `policy_2025.md` and ask:

> What is the Cyber Liability sublimit and deductible?

Show the answer and expand the citation QA panel. Point out the evidence ID and numeric/date support.

## 35–55s — Cross-document reasoning

Upload `policy_2026.md` and ask:

> Compare the Cyber Liability and Business Interruption changes between the 2025 and 2026 policies.

Show that the system routes to whole-document comparison and cites both documents.

## 55–70s — Abstention

Ask:

> What is the company's revenue in 2030?

Show:

> I couldn't find sufficient verified evidence...

Explain that the system fails closed instead of guessing.

## 70–90s — Engineering view


- hybrid retrieval
- RRF + reranking
- LangGraph routing/fan-out
- citation verification
- evaluation benchmark
- token/latency instrumentation
