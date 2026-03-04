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
import re
import time
from dataclasses import dataclass, field

from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


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

    Supports both OpenAI and Ollama backends based on the configured
    ``LLM_PROVIDER`` in settings.  Includes retry logic and JSON repair
    for smaller / local models that may produce malformed output.

    Parameters
    ----------
    model : str, optional
        Override the LLM model name.  If *None*, a sensible default is
        chosen based on the provider (gpt-4o-mini for OpenAI, qwen2.5:7b
        for Ollama).
    """

    def __init__(self, model: str | None = None):
        self._settings = get_settings()
        self._provider = self._settings.llm_provider
        default_model = (
            self._settings.ollama_model
            if self._provider == LLMProvider.OLLAMA
            else "gpt-4o-mini"
        )
        self.model = model or default_model

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
        """Call the configured LLM and return the raw response string.

        Routes to OpenAI or Ollama based on ``self._provider``.
        Includes retry logic with exponential back-off.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self._provider == LLMProvider.OLLAMA:
                    return self._call_ollama(text)
                else:
                    return self._call_openai(text)
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(
                    "LLM call attempt %d/%d failed — retrying in %.1fs …",
                    attempt, MAX_RETRIES, RETRY_DELAY * attempt,
                )
                time.sleep(RETRY_DELAY * attempt)
        # unreachable, but keeps mypy happy
        raise RuntimeError("LLM call failed after retries")

    def _call_openai(self, text: str) -> str:
        """Call the OpenAI Chat Completions API."""
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

    def _call_ollama(self, text: str) -> str:
        """Call a local Ollama server."""
        import requests

        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n{text}"
        resp = requests.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Attempt to extract valid JSON from a potentially messy LLM response.

        Handles common issues with smaller models:
        - Markdown code fences (```json ... ```)
        - Leading/trailing text around the JSON object
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = cleaned.strip().rstrip("`")

        # Try to find the outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        return cleaned

    @staticmethod
    def _parse_response(chunk_id: str, raw: str) -> ExtractionResult:
        """Parse the LLM JSON response into an ExtractionResult.

        Includes JSON repair for malformed output from smaller models.
        """
        # First try raw, then try repaired
        data = None
        for attempt_raw in (raw, EntityExtractor._repair_json(raw)):
            try:
                data = json.loads(attempt_raw)
                break
            except json.JSONDecodeError:
                continue

        if data is None:
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
