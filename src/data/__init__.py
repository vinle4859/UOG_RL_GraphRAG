# =============================================================================
# src/data/__init__.py
# Data ingestion & preprocessing sub-package.
# =============================================================================
"""
Data Pipeline
=============
Handles data loading, chunking, and classification for ArXiv papers:

1. **Loading** (``loader.py``) — Read JSON/TXT into ``Document`` objects.
2. **Chunking** (``chunker.py``) — Split documents into token-sized chunks.
3. **Classification** (``classifier.py``) — Optional sub-field clustering.

.. note::

   ``preprocessor.py`` is **reference-only** documentation for the PDF
   extraction team.  See that file for the expected JSON schema.

Data format: papers are stored as structured JSON with metadata (title,
authors, abstract, sections) — see ``preprocessor.PaperRecord`` for schema.
"""
