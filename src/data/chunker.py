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
import re
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace

import tiktoken

from src.config import get_settings
from src.data.loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Preprocessing Helpers
# ---------------------------------------------------------------------------


def _normalize_unicode(text: str) -> str:
    """Normalise unicode to NFC form (e.g. combining accents → single chars)."""
    return unicodedata.normalize("NFC", text)


def _collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace / blank lines with a single space or newline."""
    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces / tabs on a single line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _remove_control_characters(text: str) -> str:
    """Strip non-printable control characters (except newline/tab)."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))


# Default preprocessing chain applied by ChunkingPipeline
DEFAULT_PREPROCESSORS: list[Callable[[str], str]] = [
    _remove_control_characters,
    _normalize_unicode,
    _collapse_whitespace,
]


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

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}__chunk_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    index=idx,
                    metadata={
                        "title": doc.title,
                        "doc_id": doc.doc_id,
                        **doc.metadata,
                    },
                )
            )

            start += self.chunk_size - self.chunk_overlap
            idx += 1

        logger.debug("Document %s -> %d chunks", doc.doc_id, len(chunks))
        return chunks

    def chunk_documents(
        self,
        docs: Sequence[Document],
        num_workers: int | None = None,
    ) -> list[Chunk]:
        """
        Chunk multiple documents and return a flat list.

        Parameters
        ----------
        num_workers : int, optional
            Thread-pool size for parallel chunking.  Defaults to
            ``settings.num_workers``.  Use ``1`` to disable parallelism.
        """
        workers = num_workers if num_workers is not None else get_settings().num_workers

        if workers <= 1 or len(docs) <= 1:
            # Fast path — no thread overhead for tiny workloads
            all_chunks: list[Chunk] = []
            for doc in docs:
                all_chunks.extend(self.chunk_document(doc))
            logger.info("Chunked %d documents -> %d chunks", len(docs), len(all_chunks))
            return all_chunks

        # Parallel path — tiktoken.encode is thread-safe
        results: dict[int, list[Chunk]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(self.chunk_document, doc): i for i, doc in enumerate(docs)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Chunking failed for doc index %d", idx)
                    results[idx] = []

        # Preserve original document order
        all_chunks = [chunk for i in range(len(docs)) for chunk in results.get(i, [])]
        logger.info(
            "Chunked %d documents -> %d chunks (workers=%d)", len(docs), len(all_chunks), workers
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Chunking Pipeline
# ---------------------------------------------------------------------------


class ChunkingPipeline:
    """
    A configurable preprocessing + chunking pipeline.

    Applies a sequence of text-cleaning steps to each document before
    handing it off to :class:`TokenChunker`.  This is the recommended
    entry-point for building chunks from raw documents.

    Pipeline stages
    ---------------
    1. **Remove control characters** — strips non-printable bytes except
       newline and tab.
    2. **Unicode normalisation** — NFC form so combining characters are
       collapsed to single code-points.
    3. **Whitespace collapse** — reduces runs of blank lines / spaces to
       a single separator.
    4. **Token-aware chunking** — splits the cleaned text into overlapping
       token windows via :class:`TokenChunker`.

    Parameters
    ----------
    chunker : TokenChunker, optional
        Custom chunker.  Defaults to a :class:`TokenChunker` built from
        settings.
    preprocessors : list of callables, optional
        Text-to-text functions applied in order before chunking.
        Defaults to ``DEFAULT_PREPROCESSORS``.

    Example
    -------
    ::

        from src.data.loader import load_documents
        from src.data.chunker import ChunkingPipeline

        pipeline = ChunkingPipeline()
        docs = load_documents()
        chunks = pipeline.run(docs)
    """

    def __init__(
        self,
        chunker: TokenChunker | None = None,
        preprocessors: list[Callable[[str], str]] | None = None,
    ):
        self.chunker = chunker or TokenChunker()
        self.preprocessors: list[Callable[[str], str]] = (
            preprocessors if preprocessors is not None else list(DEFAULT_PREPROCESSORS)
        )

    def preprocess_text(self, text: str) -> str:
        """Apply every preprocessing step in sequence and return clean text."""
        for fn in self.preprocessors:
            text = fn(text)
        return text

    def preprocess_document(self, doc: Document) -> Document:
        """Return a new :class:`Document` with its ``text`` field cleaned."""
        cleaned = self.preprocess_text(doc.text)
        return dataclass_replace(doc, text=cleaned)

    def run(self, docs: Sequence[Document]) -> list[Chunk]:
        """
        Full pipeline: preprocess all documents then chunk them.

        Parameters
        ----------
        docs : sequence of Document
            Raw documents from :func:`~src.data.loader.load_documents`.

        Returns
        -------
        list[Chunk]
            Flat list of chunks ready for embedding.
        """
        logger.info("ChunkingPipeline: preprocessing %d documents …", len(docs))
        workers = get_settings().num_workers
        if workers > 1 and len(docs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                cleaned_docs = list(pool.map(self.preprocess_document, docs))
        else:
            cleaned_docs = [self.preprocess_document(doc) for doc in docs]
        chunks = self.chunker.chunk_documents(cleaned_docs, num_workers=workers)
        logger.info("ChunkingPipeline: produced %d chunks.", len(chunks))
        return chunks
