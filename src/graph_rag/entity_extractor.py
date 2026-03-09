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
- [ ] Support batch extraction for Ollama / vLLM.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import LLMProvider, get_settings
from src.utils.retry import openai_retry

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
- If the text consists primarily of mathematical equations with no extractable
  named entities, return exactly: {"entities": [], "relations": []}
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
            self._settings.ollama_model if self._provider == LLMProvider.OLLAMA else "gpt-4o-mini"
        )
        self.model = model or default_model

    # Minimum fraction of tokens that must look like natural language
    # before we bother calling the LLM.
    _MATH_PASS_THRESHOLD: float = 0.5

    # Max concurrent Ollama requests.  2-3 works well for a 0.8B model on
    # a 4 GB VRAM GPU; increase if VRAM headroom allows.
    _DEFAULT_OLLAMA_WORKERS: int = 1  # Ollama is sequential on GPU; parallelism stacks wait time

    def extract(self, chunk_id: str, text: str) -> ExtractionResult:
        """
        Extract entities and relations from a single text chunk.

        Math expressions are stripped from *text* before the LLM call so that
        inline equations don't corrupt the JSON output.  If stripping leaves
        fewer than 20 word-like tokens the chunk is still skipped (it was
        essentially pure math with no useful prose).

        Returns
        -------
        ExtractionResult
        """
        cleaned = self._strip_math(text)
        tokens = cleaned.split()
        word_tokens = sum(1 for t in tokens if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.60)
        if word_tokens < 20:
            logger.debug("Skipping near-empty chunk %s after math stripping", chunk_id)
            return ExtractionResult(chunk_id=chunk_id)

        raw_json = self._call_llm(cleaned)
        return self._parse_response(chunk_id, raw_json)

    def extract_batch(
        self,
        chunks: list[tuple[str, str]],
        checkpoint_path: Path | str | None = None,
        num_workers: int | None = None,
    ) -> list[ExtractionResult]:
        """
        Extract from multiple (chunk_id, text) pairs.

        Supports checkpointing: results are appended to *checkpoint_path* (JSONL)
        after each chunk so a crashed run can be resumed without re-processing
        already-completed chunks.

        Concurrent Ollama requests are issued via a thread pool so GPU
        utilisation stays high.  *num_workers* defaults to
        ``_DEFAULT_OLLAMA_WORKERS`` (3) for Ollama and 1 for OpenAI
        (rate-limited API).

        Parameters
        ----------
        chunks : list of (chunk_id, text)
        checkpoint_path : Path, optional
            JSONL file to read existing results from and append new results to.
            Pass the same path on every run to get resume behaviour.
        num_workers : int, optional
            Number of concurrent LLM requests.  Defaults to 3 for Ollama.
        """
        from tqdm import tqdm

        workers = (
            num_workers
            if num_workers is not None
            else (self._DEFAULT_OLLAMA_WORKERS if self._provider == LLMProvider.OLLAMA else 1)
        )

        # --- Load checkpoint ---
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        done: dict[str, ExtractionResult] = {}
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    with contextlib.suppress(Exception):
                        rec = self._deserialize(json.loads(line))
                        done[rec.chunk_id] = rec
            logger.info(
                "Checkpoint: %d/%d chunks already extracted — skipping.",
                len(done),
                len(chunks),
            )

        # --- Open checkpoint file for appending (thread-safe via lock) ---
        ckpt_fh = None
        ckpt_lock = threading.Lock()
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_fh = open(checkpoint_path, "a", encoding="utf-8")  # noqa: SIM115

        def _write_checkpoint(result: ExtractionResult) -> None:
            if ckpt_fh:
                with ckpt_lock:
                    ckpt_fh.write(json.dumps(asdict(result)) + "\n")
                    ckpt_fh.flush()

        # --- Process pending chunks ---
        results: list[ExtractionResult] = list(done.values())
        pending = [(cid, txt) for cid, txt in chunks if cid not in done]

        def _extract_one(args: tuple[str, str]) -> ExtractionResult:
            cid, txt = args
            try:
                return self.extract(cid, txt)
            except Exception:
                logger.exception("Extraction failed for chunk %s", cid)
                return ExtractionResult(chunk_id=cid)

        try:
            pbar = tqdm(total=len(pending), desc="Extracting entities", unit="chunk")
            if workers <= 1:
                # Sequential path
                for item in pending:
                    result = _extract_one(item)
                    results.append(result)
                    _write_checkpoint(result)
                    pbar.update(1)
            else:
                # Parallel path — concurrent Ollama requests
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    future_to_id = {pool.submit(_extract_one, item): item[0] for item in pending}
                    for future in as_completed(future_to_id):
                        result = future.result()
                        results.append(result)
                        _write_checkpoint(result)
                        pbar.update(1)
            pbar.close()
        finally:
            if ckpt_fh:
                ckpt_fh.close()

        return results

    @staticmethod
    def _deserialize(data: dict) -> ExtractionResult:
        """Reconstruct an ExtractionResult from a checkpoint dict."""
        return ExtractionResult(
            chunk_id=data["chunk_id"],
            entities=[Entity(**e) for e in data.get("entities", [])],
            relations=[Relation(**r) for r in data.get("relations", [])],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_math(text: str) -> str:
        """Remove mathematical expressions from *text*, preserving prose.

        Stripping order (each step operates on the result of the previous):

        1. LaTeX display environments  ``\\begin{...} ... \\end{...}``
        2. Display math delimiters     ``$$ ... $$``  and  ``\\[ ... \\]``
        3. Inline math delimiters      ``$ ... $``    and  ``\\( ... \\)``
        4. Bare LaTeX commands         ``\\cmd{...}{...}`` (any braced args)
        5. Tokens whose characters are >60 % non-alphabetic (stray symbols)
        6. Collapse runs of whitespace left by removed segments
        """
        # 1 — LaTeX environments
        text = re.sub(r"\\begin\s*\{[^}]*\}.*?\\end\s*\{[^}]*\}", " ", text, flags=re.DOTALL)

        # 2 — display math
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)

        # 3 — inline math
        text = re.sub(r"\$[^$\n]{1,300}?\$", " ", text)
        text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)

        # 4 — bare LaTeX commands with braced arguments  e.g. \frac{a}{b}
        # Repeat up to 4 times to handle nested braces
        _latex_cmd = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^{}]*\})+")
        for _ in range(4):
            text, n = _latex_cmd.subn(" ", text)
            if n == 0:
                break

        # 5 — stray symbol tokens (e.g. "=x^2", "_{k}", "≤0.05")
        tokens = text.split()
        tokens = [t for t in tokens if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.40]
        text = " ".join(tokens)

        # 6 — normalise whitespace
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text

    @staticmethod
    def _is_math_heavy(text: str) -> bool:
        """Return True when *text* is dominated by mathematical notation.

        Used as a diagnostic / logging helper.  The active guard in
        :meth:`extract` operates on the *stripped* text instead.
        """
        if re.search(r"\\begin\s*\{", text):
            return True
        math_chars = set(r"+=<>^_\|∑∫∂∈≤≥≠→∞±×÷√θλσμπαβγδεζη")
        math_char_ratio = sum(1 for c in text if c in math_chars) / max(len(text), 1)
        if math_char_ratio > 0.20:
            return True
        tokens = text.split()
        if not tokens:
            return False
        word_tokens = sum(1 for t in tokens if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.60)
        return (word_tokens / len(tokens)) < EntityExtractor._MATH_PASS_THRESHOLD

    @openai_retry()
    def _call_llm(self, text: str) -> str:
        """Call the configured LLM and return the raw response string.

        Routes to OpenAI or Ollama based on ``self._provider``.
        Decorated with :func:`~src.utils.retry.openai_retry` so transient
        rate-limit / quota errors are retried with exponential back-off.
        """
        if self._provider == LLMProvider.OLLAMA:
            return self._call_ollama(text)
        return self._call_openai(text)

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
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Ollama JSON mode: constrains output to valid JSON
                "options": {
                    "num_ctx": 2048,
                    "num_predict": 1024,  # room for verbose entity lists
                    "temperature": 0,
                },
            },
            timeout=180,  # format:json constrained decoding is slower; allow for it
        )
        resp.raise_for_status()
        return resp.json()["response"]

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Attempt to extract valid JSON from a potentially messy LLM response.

        Handles common issues with smaller models:
        - Markdown code fences (```json ... ```)
        - Leading/trailing text around the JSON object
        - Response wrapped in a JSON array instead of an object
        - Qwen3 <think>...</think> reasoning preamble
        """
        # Strip Qwen3 thinking blocks before anything else
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.strip().rstrip("`").strip()

        # If the model wrapped the object in an array, unwrap the first element
        if cleaned.startswith("["):
            inner_start = cleaned.find("{")
            inner_end = cleaned.rfind("}")
            if inner_start != -1 and inner_end != -1:
                cleaned = cleaned[inner_start : inner_end + 1]
                return cleaned

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
        Handles common small-model quirks:
        - Markdown fences around the JSON
        - entities/relations as lists of strings instead of dicts
        - Unexpected nesting (e.g. {"entities": {"list": [...]}})
        """
        # First try raw, then try repaired
        data = None
        for attempt_raw in (raw, EntityExtractor._repair_json(raw)):
            try:
                parsed = json.loads(attempt_raw)
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except json.JSONDecodeError:
                continue

        if data is None:
            logger.warning("Invalid JSON from LLM for chunk %s — skipping.", chunk_id)
            return ExtractionResult(chunk_id=chunk_id)

        # Normalise entities: accept list-of-dicts OR list-of-strings
        raw_entities = data.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        entities: list[Entity] = []
        for e in raw_entities:
            if isinstance(e, dict):
                entities.append(
                    Entity(
                        name=e.get("name", ""),
                        entity_type=e.get("type", e.get("entity_type", "UNKNOWN")),
                        description=e.get("description", ""),
                    )
                )
            elif isinstance(e, str) and e.strip():
                # Small model returned plain string names — wrap them
                entities.append(Entity(name=e.strip(), entity_type="UNKNOWN"))

        # Normalise relations: accept list-of-dicts OR list-of-strings
        raw_relations = data.get("relations", [])
        if not isinstance(raw_relations, list):
            raw_relations = []

        relations: list[Relation] = []
        for r in raw_relations:
            if isinstance(r, dict):
                relations.append(
                    Relation(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        relation_type=r.get("relation", r.get("relation_type", "")),
                        description=r.get("description", ""),
                        chunk_id=chunk_id,
                    )
                )
            # Plain-string relations can't be meaningfully parsed — skip silently

        return ExtractionResult(chunk_id=chunk_id, entities=entities, relations=relations)
