# =============================================================================
# src/rag/__init__.py
# Standard RAG pipeline sub-package.
# =============================================================================
"""
Standard RAG Pipeline
=====================
Implements the classic Retrieve-Augmented Generation flow:

1. **Index** — Embed document chunks → store in a vector database.
2. **Retrieve** — Given a query, find the top-k most similar chunks.
3. **Generate** — Feed retrieved context + query to an LLM for the answer.

Modules
-------
- ``embedder``   — Embedding model wrappers (OpenAI, Sentence-Transformers).
- ``vectorstore`` — Vector DB abstraction (ChromaDB, FAISS).
- ``retriever``  — Top-k retrieval logic.
- ``generator``  — LLM answer generation with context.
- ``pipeline``   — End-to-end orchestration.
"""
