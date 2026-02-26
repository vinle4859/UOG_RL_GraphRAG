# =============================================================================
# src/evaluation/__init__.py
# Benchmarking & evaluation sub-package.
# =============================================================================
"""
Evaluation & Benchmarking
=========================
Compares standard RAG vs GraphRAG along multiple dimensions:

- **Retrieval quality** — Precision@k, Recall@k, MRR.
- **Answer quality** — ROUGE, BERTScore, LLM-as-judge.
- **Efficiency** — Latency, token usage, cost.

Modules
-------
- ``metrics``    — Metric computation functions.
- ``benchmark``  — Orchestrate side-by-side evaluation runs.
- ``questions``  — Manage evaluation question sets.
"""
