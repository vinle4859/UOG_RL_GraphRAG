# =============================================================================
# src/data/loader.py
# Load papers from structured JSON (preferred) or legacy .txt files.
# =============================================================================
"""
Document Loader
===============
Loads papers into a uniform ``Document`` schema for downstream modules
(RAG, GraphRAG).

Supports two input formats (auto-detected):

1. **Structured JSON** (preferred) — one ``.json`` per paper with title,
   authors, abstract, and full text.  Your team's PDF extraction pipeline
   should produce files in this format (see ``preprocessor.py`` for the
   reference schema).
2. **Legacy plain .txt** — fallback if no JSON files are found.

Typical workflow::

    # Place JSON or TXT files in data/processed/
    from src.data.loader import load_documents, search_by_title
    docs = load_documents()                           # loads from processed/
    matches = search_by_title("reinforcement learning")  # title search
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    A single ArXiv paper with metadata and text.

    Attributes
    ----------
    doc_id : str
        Unique identifier (typically the ArXiv ID or filename stem).
    text : str
        Full-text content used for chunking and embedding.
    title : str
        Paper title (empty if unknown).
    abstract : str
        Paper abstract (empty if unknown).
    authors : list[str]
        Author names.
    categories : list[str]
        ArXiv categories (e.g. ["cs.LG", "cs.AI"]).
    year : int | None
        Publication year.
    metadata : dict
        Any additional metadata (sections, source file, etc.).
    """
    doc_id: str
    text: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    year: int | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def iter_documents(data_dir: Path | None = None) -> Iterator[Document]:
    """
    Yield :class:`Document` objects from *data_dir*.

    **Auto-detection logic:**

    1. Look for ``.json`` files in ``processed_data_dir`` first (structured).
    2. If none found, fall back to ``.txt`` files in ``raw_data_dir``.

    Parameters
    ----------
    data_dir : Path, optional
        Explicit directory to load from.  If omitted, tries
        ``processed_data_dir`` then ``raw_data_dir`` from settings.

    Yields
    ------
    Document
        One per paper.
    """
    settings = get_settings()

    if data_dir is not None:
        data_dir = Path(data_dir)
    else:
        # Prefer structured JSON from processed/
        processed = Path(settings.processed_data_dir)
        raw = Path(settings.raw_data_dir)

        json_files = sorted(processed.glob("*.json")) if processed.exists() else []
        if json_files:
            logger.info("Loading %d structured JSON files from %s", len(json_files), processed)
            yield from _load_json_files(json_files)
            return

        # Fallback to raw .txt
        if raw.exists():
            txt_files = sorted(raw.glob("*.txt"))
            if txt_files:
                logger.info(
                    "No JSON files in %s — falling back to %d .txt files in %s",
                    processed, len(txt_files), raw,
                )
                yield from _load_txt_files(txt_files)
                return

        logger.warning("No data found in %s or %s", processed, raw)
        return

    # Explicit directory: try JSON first, then txt
    json_files = sorted(data_dir.glob("*.json"))
    if json_files:
        yield from _load_json_files(json_files)
        return

    txt_files = sorted(data_dir.glob("*.txt"))
    if txt_files:
        yield from _load_txt_files(txt_files)
        return

    logger.warning("No .json or .txt files found in %s", data_dir)


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """Load all documents into a list (convenience wrapper)."""
    return list(iter_documents(data_dir))


# ---------------------------------------------------------------------------
# Title / Metadata Search
# ---------------------------------------------------------------------------

def search_by_title(
    query: str,
    documents: Sequence[Document] | None = None,
    top_n: int = 10,
) -> list[Document]:
    """
    Search papers by title (case-insensitive substring match).

    Parameters
    ----------
    query : str
        Search string (e.g. "reinforcement learning").
    documents : Sequence[Document], optional
        Pre-loaded documents.  If None, loads from default directory.
    top_n : int
        Maximum number of results to return.

    Returns
    -------
    list[Document]
        Papers whose title contains *query*, sorted by relevance (exact
        match first, then by position of match).
    """
    if documents is None:
        documents = load_documents()

    query_lower = query.lower()
    matches: list[tuple[int, Document]] = []

    for doc in documents:
        title_lower = doc.title.lower()
        if query_lower in title_lower:
            # Score: earlier match position = more relevant
            pos = title_lower.index(query_lower)
            matches.append((pos, doc))

    matches.sort(key=lambda x: x[0])
    return [doc for _, doc in matches[:top_n]]


def search_by_metadata(
    documents: Sequence[Document],
    *,
    year: int | None = None,
    category: str | None = None,
    author: str | None = None,
) -> list[Document]:
    """
    Filter papers by metadata fields.

    Parameters
    ----------
    documents : Sequence[Document]
        Papers to filter.
    year : int, optional
        Filter by publication year.
    category : str, optional
        Filter by ArXiv category (substring match).
    author : str, optional
        Filter by author name (case-insensitive substring).

    Returns
    -------
    list[Document]
        Papers matching ALL specified criteria.
    """
    results = list(documents)

    if year is not None:
        results = [d for d in results if d.year == year]

    if category is not None:
        cat_lower = category.lower()
        results = [d for d in results if any(cat_lower in c.lower() for c in d.categories)]

    if author is not None:
        author_lower = author.lower()
        results = [d for d in results if any(author_lower in a.lower() for a in d.authors)]

    return results


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _load_json_files(files: list[Path]) -> Iterator[Document]:
    """Load Documents from structured JSON files."""
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            yield Document(
                doc_id=data.get("doc_id", fpath.stem),
                text=data.get("full_text", ""),
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                authors=data.get("authors", []),
                categories=data.get("categories", []),
                year=data.get("year"),
                metadata={
                    "sections": data.get("sections", []),
                    "source_file": data.get("source_file", ""),
                },
            )
        except Exception:
            logger.exception("Failed to load JSON: %s", fpath.name)


def _load_txt_files(files: list[Path]) -> Iterator[Document]:
    """Load Documents from plain .txt files (legacy fallback)."""
    for fpath in files:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        # Heuristic: first non-empty line is the title
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        title = lines[0] if lines else ""

        yield Document(doc_id=fpath.stem, text=text, title=title)
