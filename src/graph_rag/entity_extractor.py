# =============================================================================
# src/graph_rag/entity_extractor.py
# Extract entities and relationships from text using an LLM.
# =============================================================================
"""
Entity & Relation Extraction
=============================
Uses a structured LLM prompt to extract (subject, predicate, object) triples
from document chunks.  These triples form the edges of the knowledge graph.

Design Notes
------------
- We instruct the LLM to return **JSON** so parsing is reliable.
- Entity types can be constrained to a predefined schema (e.g., Method,
  Dataset, Metric, Task) or left open.
- Extraction runs per-chunk to keep context within the LLM's window.

TODO
----
- [ ] Add caching so re-extraction on the same chunk is free.
- [ ] Support batch extraction for Ollama / vLLM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from text."""
    name: str
    entity_type: str = "UNKNOWN"
    description: str = ""


@dataclass
class Relation:
    """A directed relationship between two entities."""
    source: str
    target: str
    relation_type: str
    description: str = ""
    chunk_id: str = ""


@dataclass
class ExtractionResult:
    """Entities + relations extracted from a single chunk."""
    chunk_id: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert information extraction system.
Given a passage of text from a research paper, extract all significant
entities and the relationships between them.

Return a JSON object with two keys:
  "entities": [ {"name": "...", "type": "...", "description": "..."}, ... ]
  "relations": [ {"source": "...", "target": "...", "relation": "...", "description": "..."}, ... ]

Entity types to consider: Method, Algorithm, Dataset, Metric, Task, Model, \
Framework, Concept, Person, Organisation.

Rules:
- Normalise entity names (e.g. "reinforcement learning" → "Reinforcement Learning").
- Use concise relation labels (e.g. "uses", "outperforms", "is_variant_of").
- Only output valid JSON — no markdown fences, no extra text.
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Extract entities and relations from text chunks via an LLM.

    Parameters
    ----------
    model : str
        LLM model name (e.g. "gpt-4o-mini").
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._settings = get_settings()

    def extract(self, chunk_id: str, text: str) -> ExtractionResult:
        """
        Extract entities and relations from a single text chunk.

        Returns
        -------
        ExtractionResult
        """
        raw_json = self._call_llm(text)
        return self._parse_response(chunk_id, raw_json)

    def extract_batch(self, chunks: list[tuple[str, str]]) -> list[ExtractionResult]:
        """
        Extract from multiple (chunk_id, text) pairs.

        TODO: Parallelise with asyncio / thread pool for speed.
        """
        results = []
        for chunk_id, text in chunks:
            try:
                results.append(self.extract(chunk_id, text))
            except Exception:
                logger.exception("Extraction failed for chunk %s", chunk_id)
                results.append(ExtractionResult(chunk_id=chunk_id))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, text: str) -> str:
        """Call the configured LLM and return the raw response string."""
        from openai import OpenAI

        client = OpenAI(api_key=self._settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_response(chunk_id: str, raw: str) -> ExtractionResult:
        """Parse the LLM JSON response into an ExtractionResult."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from LLM for chunk %s — skipping.", chunk_id)
            return ExtractionResult(chunk_id=chunk_id)

        entities = [
            Entity(
                name=e.get("name", ""),
                entity_type=e.get("type", "UNKNOWN"),
                description=e.get("description", ""),
            )
            for e in data.get("entities", [])
        ]
        relations = [
            Relation(
                source=r.get("source", ""),
                target=r.get("target", ""),
                relation_type=r.get("relation", ""),
                description=r.get("description", ""),
                chunk_id=chunk_id,
            )
            for r in data.get("relations", [])
        ]
        return ExtractionResult(chunk_id=chunk_id, entities=entities, relations=relations)
