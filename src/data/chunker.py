# =============================================================================
# src/data/chunker.py
# Split documents into overlapping chunks suitable for embedding.
# =============================================================================
"""
Text Chunking
=============
Provides token-aware chunking strategies.  The default uses ``tiktoken``
for accurate GPT-compatible token counts, but also supports a simple
character-based fallback.

Design decisions
----------------
* **Token-based chunking** is preferred because embedding models and LLMs
  have *token* limits, not character limits.
* **Overlap** ensures that sentences straddling chunk boundaries are not
  lost.  The default overlap is ~12 % of chunk size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import tiktoken

from src.config import get_settings
from src.data.loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A contiguous text fragment of a parent document.

    Attributes
    ----------
    chunk_id : str
        ``{doc_id}__chunk_{index}`` for traceability.
    doc_id : str
        Parent document identifier.
    text : str
        The chunk content.
    index : int
        0-based ordinal within the parent document.
    metadata : dict
        Inherited + chunk-specific metadata.
    """
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TokenChunker:
    """
    Split text into chunks of approximately ``chunk_size`` tokens with
    ``chunk_overlap`` token overlap.

    Parameters
    ----------
    chunk_size : int
        Target number of tokens per chunk.
    chunk_overlap : int
        Number of overlapping tokens between consecutive chunks.
    encoding_name : str
        Tiktoken encoding (default ``cl100k_base`` — used by GPT-4 / ada-002).
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single :class:`Document` into :class:`Chunk` objects."""
        tokens = self.encoder.encode(doc.text)
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}__chunk_{idx}",
                doc_id=doc.doc_id,
                text=chunk_text,
                index=idx,
                metadata={
                    "title": doc.title,
                    "doc_id": doc.doc_id,
                    **doc.metadata,
                },
            ))

            start += self.chunk_size - self.chunk_overlap
            idx += 1

        logger.debug("Document %s → %d chunks", doc.doc_id, len(chunks))
        return chunks

    def chunk_documents(self, docs: Sequence[Document]) -> list[Chunk]:
        """Chunk multiple documents and return a flat list."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Chunked %d documents → %d chunks", len(docs), len(all_chunks))
        return all_chunks
