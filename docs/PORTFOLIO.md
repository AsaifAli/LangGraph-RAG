# Portfolio Positioning

## One-line pitch

**Evidence-Grounded Sparse-First Agentic RAG Platform** — a LangGraph research system combining BM25/exact/phrase/fuzzy OpenSearch retrieval, optional neural sparse retrieval, bounded agentic query expansion, cross-encoder reranking, web research, persistent memory, and fail-closed citation QA.

## What to demonstrate

1. **Grounded answer:** ask for a policy fact and open the citation evidence.
2. **Semantic recovery:** use a paraphrased question whose wording differs from the source and show neural-sparse/agentic query expansion recovering the evidence.
3. **Cross-document comparison:** compare two policy documents.
4. **Abstention:** ask for a fact absent from the documents and show the system refusing to invent it.
5. **Performance:** show retrieval latency, candidate counts, rerank latency, token usage, and retrieval strategy in the benchmark report.

## Why it stands out

The project is not a generic chatbot and is not tied to a single vector database. Its retrieval layer is an explicit engineering experiment: classical lexical search for transparency and exact matching, neural sparse retrieval for semantic recall without dense k-NN, bounded agentic query reformulation for hard queries, and deterministic evidence verification before synthesis.

## Recruiter talking points

- Why sparse-first retrieval instead of dense-only search?
- What semantic gap does neural sparse retrieval close?
- Why fuse lexical and neural-sparse candidates with RRF?
- When does the agent reformulate a query, and how is runaway search prevented?
- Why rerank only a small candidate set?
- How are citations authorized and verified?
- How would you evaluate retrieval independently from generation?
