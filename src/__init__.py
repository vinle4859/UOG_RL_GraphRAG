# =============================================================================
# src/__init__.py
# Root package for the UOG RL GraphRAG benchmarking project.
# =============================================================================
"""
UOG RL GraphRAG — Benchmarking GraphRAG vs Standard RAG
========================================================

This package contains all pipeline code for:
- Data ingestion & preprocessing (text files from ArXiv papers)
- Standard RAG pipeline (embed → retrieve → generate)
- Graph RAG pipeline (extract entities → build graph → community detection → retrieve → generate)
- Evaluation & benchmarking utilities

See README.md for setup and usage instructions.
"""

__version__ = "0.1.0"
