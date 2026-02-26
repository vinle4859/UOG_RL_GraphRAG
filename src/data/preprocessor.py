# =============================================================================
# src/data/preprocessor.py
# REFERENCE ONLY — PDF extraction is handled by a separate team member.
# This module is kept as documentation of the expected JSON schema and
# heuristic extraction logic.  It is NOT imported by the active pipeline.
# =============================================================================
"""
PDF Preprocessing Pipeline  (REFERENCE / DOCUMENTATION ONLY)
=============================================================
Converts raw ArXiv PDF files into structured JSON records with extracted
metadata (title, abstract, authors, sections) and clean full text.

.. note::

   This file is **not used** by the main RAG / GraphRAG pipeline.
   PDF extraction is owned by a separate team member.  The code is
   retained so that the extraction engineer can see the target JSON
   schema and reuse the heuristic helpers if desired.

Why not plain .txt?
-------------------
Plain ``.txt`` loses all structure — you can't search by title, filter by
author, or separate the abstract from the body.  Structured **JSON** keeps
the metadata alongside the text, enabling:

- **Title search** — find papers by name instantly.
- **Abstract-only RAG** — sometimes you only need the summary.
- **Section-aware chunking** — split on section boundaries for better chunks.
- **Filtered retrieval** — e.g. "only papers about reinforcement learning".

Supported Input Formats
-----------------------
- ``.pdf``  — Uses PyMuPDF (fitz) for extraction (fast, accurate).
- ``.txt``  — Falls back to plain text with minimal metadata.

Output Format
-------------
Each paper becomes a JSON file::

    {
      "doc_id": "2301.00001",
      "title": "Proximal Policy Optimization Algorithms",
      "authors": ["John Schulman", "..."],
      "abstract": "We propose a new family of ...",
      "categories": ["cs.LG", "cs.AI"],
      "year": 2023,
      "full_text": "1 Introduction\\nReinforcement learning ...",
      "sections": [
        {"heading": "Introduction", "text": "..."},
        {"heading": "Method", "text": "..."}
      ],
      "source_file": "2301.00001.pdf"
    }

Dependencies
------------
::

    pip install PyMuPDF   # imported as 'fitz'
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class PaperRecord:
    """
    Structured representation of an academic paper.

    This is the canonical storage format — everything downstream (loader,
    chunker, RAG, GraphRAG) reads from this.
    """
    doc_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    year: int | None = None
    full_text: str = ""
    sections: list[dict] = field(default_factory=list)  # [{"heading": ..., "text": ...}]
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaperRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_path: Path) -> PaperRecord:
    """
    Extract text and metadata from a single PDF file.

    Uses PyMuPDF (fitz) for fast, accurate text extraction.  Attempts
    heuristic parsing of title, abstract, and section headings.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    PaperRecord
        Structured record with extracted metadata and text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. Install it with:\n"
            "  pip install PyMuPDF"
        )

    doc_id = pdf_path.stem
    logger.info("Extracting PDF: %s", pdf_path.name)

    pdf = fitz.open(str(pdf_path))
    pages_text: list[str] = []
    for page in pdf:
        pages_text.append(page.get_text("text"))
    pdf.close()

    full_text = "\n".join(pages_text)

    # --- Heuristic metadata extraction ---
    title = _extract_title(pages_text[0] if pages_text else "")
    abstract = _extract_abstract(full_text)
    authors = _extract_authors(pages_text[0] if pages_text else "")
    sections = _extract_sections(full_text)
    year = _extract_year(doc_id, full_text)

    # Clean the full text
    full_text = _clean_text(full_text)

    return PaperRecord(
        doc_id=doc_id,
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        full_text=full_text,
        sections=sections,
        source_file=pdf_path.name,
    )


def extract_from_txt(txt_path: Path) -> PaperRecord:
    """
    Create a PaperRecord from a plain .txt file.

    Since .txt files have no inherent structure, we use heuristics to
    guess the title (first non-empty line) and abstract.

    Parameters
    ----------
    txt_path : Path
        Path to the text file.

    Returns
    -------
    PaperRecord
    """
    doc_id = txt_path.stem
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    title = _extract_title(text)
    abstract = _extract_abstract(text)
    full_text = _clean_text(text)

    return PaperRecord(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        full_text=full_text,
        source_file=txt_path.name,
    )


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def preprocess_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    file_types: tuple[str, ...] = (".pdf", ".txt"),
) -> list[Path]:
    """
    Process all papers in *input_dir* and write structured JSON to *output_dir*.

    Parameters
    ----------
    input_dir : Path
        Directory containing raw PDF or TXT files.
    output_dir : Path
        Directory to write JSON files.
    file_types : tuple
        Which file extensions to process.

    Returns
    -------
    list[Path]
        Paths to the generated JSON files.

    Example
    -------
    ::

        from src.data.preprocessor import preprocess_directory

        # Convert all PDFs in data/raw/ → JSON in data/processed/
        preprocess_directory("data/raw", "data/processed")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in file_types and f.is_file()
    )
    logger.info("Found %d files to preprocess in %s", len(files), input_dir)

    output_paths: list[Path] = []
    for fpath in files:
        try:
            if fpath.suffix.lower() == ".pdf":
                record = extract_from_pdf(fpath)
            else:
                record = extract_from_txt(fpath)

            out_path = output_dir / f"{record.doc_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

            output_paths.append(out_path)
            logger.debug("Wrote %s", out_path.name)

        except Exception:
            logger.exception("Failed to process %s — skipping.", fpath.name)

    logger.info("Preprocessed %d / %d files → %s", len(output_paths), len(files), output_dir)
    return output_paths


def iter_paper_records(processed_dir: Path | str) -> Iterator[PaperRecord]:
    """
    Yield PaperRecord objects from all ``.json`` files in *processed_dir*.
    """
    processed_dir = Path(processed_dir)
    for jpath in sorted(processed_dir.glob("*.json")):
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        yield PaperRecord.from_dict(data)


# ---------------------------------------------------------------------------
# Heuristic Helpers (private)
# ---------------------------------------------------------------------------

def _extract_title(first_page: str) -> str:
    """
    Guess the title from the first page text.

    Heuristic: the title is usually the first substantial line(s) of text,
    often in a larger font (which appears first in extraction order).
    """
    lines = [ln.strip() for ln in first_page.split("\n") if ln.strip()]
    if not lines:
        return ""

    # Skip lines that look like ArXiv headers ("arXiv:2301.00001v1 [cs.LG]")
    title_lines = []
    for line in lines[:5]:  # title is rarely past line 5
        if re.match(r"(arXiv|preprint|submitted|published|copyright)", line, re.IGNORECASE):
            continue
        if len(line) < 5:
            continue
        title_lines.append(line)
        # Stop if the next line looks like an author list (contains commas + short words)
        if len(title_lines) >= 2:
            break

    return " ".join(title_lines).strip()


def _extract_abstract(text: str) -> str:
    """
    Extract the abstract from paper text.

    Looks for text between "Abstract" and the first section heading.
    """
    # Try explicit "Abstract" marker
    match = re.search(
        r"(?:^|\n)\s*Abstract[:\s]*\n?(.*?)(?:\n\s*(?:1[\.\s]|I[\.\s]|Introduction|Keywords))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()
        # Clean up line breaks within the abstract
        abstract = re.sub(r"\s*\n\s*", " ", abstract)
        return abstract[:2000]  # cap length

    return ""


def _extract_authors(first_page: str) -> list[str]:
    """
    Heuristic: look for lines with comma-separated names after the title.

    This is imperfect — a proper solution would use GROBID or the ArXiv API.
    """
    lines = [ln.strip() for ln in first_page.split("\n") if ln.strip()]
    for i, line in enumerate(lines[1:6], start=1):  # skip title (line 0)
        # Author lines often have commas and no digits (unlike affiliations with zip codes)
        if "," in line and not re.search(r"\d{4,}", line):
            # Split on comma or "and"
            names = re.split(r",\s*|\s+and\s+", line)
            names = [n.strip() for n in names if 2 < len(n.strip()) < 50]
            if 1 <= len(names) <= 20:
                return names
    return []


def _extract_sections(text: str) -> list[dict]:
    """
    Split text into sections by detecting headings like "1 Introduction",
    "2. Related Work", "III. Method", etc.
    """
    # Pattern: number/roman numeral + optional period + heading text
    pattern = r"\n\s*(?:\d+\.?\s+|[IVX]+\.?\s+)([A-Z][^\n]{2,60})\n"
    matches = list(re.finditer(pattern, text))

    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({"heading": heading, "text": section_text[:5000]})

    return sections


def _extract_year(doc_id: str, text: str) -> int | None:
    """Try to extract the publication year from the ArXiv ID or text."""
    # ArXiv IDs: "2301.00001" → year 2023
    match = re.match(r"(\d{2})\d{2}\.", doc_id)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy

    # Fallback: look for a 4-digit year in text
    match = re.search(r"20[12]\d", text[:500])
    if match:
        return int(match.group())
    return None


def _clean_text(text: str) -> str:
    """Light cleaning: collapse whitespace, remove page headers/footers."""
    # Remove common PDF artifacts
    text = re.sub(r"\f", "\n", text)  # form feeds
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
    return text.strip()
