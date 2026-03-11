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
    """Entities + relations extracted from a single chunk.

    ``status`` values:
    - ``"ok"``               — extracted successfully, schema valid
    - ``"skipped_math"``    — near-pure math chunk after stripping (<20 word tokens)
    - ``"predicted_overflow"``— estimated tokens exceeded safe threshold; chunk was split
    - ``"invalid_json"``    — JSON could not be parsed even after repair
    - ``"truncated_output"``— LLM response ended before JSON was closed (length cutoff)
    - ``"schema_mismatch"`` — parsed JSON lacked required keys
    - ``"sparse_zero_entity"``— chunk was dense enough but model returned 0 entities
    - ``"fallback_success"``— primary model failed; fallback model succeeded
    - ``"fallback_fail"``   — both primary and fallback models failed
    - ``"failed"``          — unexpected exception during extraction
    """

    chunk_id: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    status: str = "ok"


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
- Use concise canonical relation labels (e.g. "uses", "outperforms", "is_variant_of",
  "trained_on", "evaluated_on", "part_of", "extends", "proposes").
- Only output valid JSON — no markdown fences, no extra text.
- If the text consists primarily of mathematical equations with no extractable
  named entities, return exactly: {"entities": [], "relations": []}
"""

REPAIR_SYSTEM_PROMPT = """\
You are a JSON extraction system. Your previous response was malformed.
Extract entities and relations from the text and return ONLY valid JSON.

Required format (nothing else — no prose, no code fences):
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relations": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]}

Entity types: Method, Algorithm, Dataset, Metric, Task, Model, Framework, Concept,
Person, Organisation.
If nothing is extractable, return: {"entities": [], "relations": []}
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
        # Fallback model used after primary + repair retry both fail.
        self._fallback_model: str | None = getattr(self._settings, "ollama_fallback_model", None)

    # Minimum fraction of tokens that must look like natural language
    # before we bother calling the LLM.
    _MATH_PASS_THRESHOLD: float = 0.5

    # Max concurrent Ollama requests.  2-3 works well for a 0.8B model on
    # a 4 GB VRAM GPU; increase if VRAM headroom allows.
    _DEFAULT_OLLAMA_WORKERS: int = 1  # Ollama is sequential on GPU; parallelism stacks wait time

    # Ollama effective context window (tokens).  Must match num_ctx below.
    _OLLAMA_NUM_CTX: int = 2048

    # Only use up to this fraction of the context budget before flagging overflow.
    _CTX_SAFE_FRACTION: float = 0.72

    # Approximate tokens added by the system prompt + formatting overhead.
    _PROMPT_OVERHEAD_TOKENS: int = 160

    # Reserved output token budget (subtracted from safe budget).
    _OUTPUT_RESERVE_TOKENS: int = 256

    # A chunk is considered "dense" (many entities expected) when it
    # has at least this many word tokens AND the capitalized-span ratio
    # is above _DENSE_CAP_RATIO.  Used by the sparse guard.
    # 0.08 (8 %) was too easy to trigger: sentence-start capitals alone
    # can reach 10-12 % in academic prose, causing many false positives on
    # notation / reference sections that legitimately have 0 extractable
    # entities.  0.18 requires genuine domain-noun density (≈1 in 5 tokens
    # is a proper noun / acronym / capitalised term beyond sentence starts).
    _DENSE_MIN_WORD_TOKENS: int = 80
    _DENSE_CAP_RATIO: float = 0.40  # ≥40 % of tokens start with uppercase

    # Characters that indicate a token is mathematical notation rather than
    # prose.  Used by _strip_math() step 6 together with the alpha-ratio
    # guard: a token is stripped only when it has < 40 % alphabetic chars
    # AND contains at least one of these characters.
    # Deliberately excludes +, <, > (appear in "C++", ">=7B", comparisons).
    _MATH_TOKEN_CHARS: frozenset[str] = frozenset("=^_\\{}~|∑∫∂∈≤≥≠→∞±×÷√θλσμπαβγδεζηρτυφχψω")

    @staticmethod
    def _is_math_token(tok: str) -> bool:
        """Return True when *tok* should be stripped as a math symbol.

        A token is considered mathematical when it is both sparsely
        alphabetic (< 40 %) *and* contains at least one character from
        :attr:`_MATH_TOKEN_CHARS`.  Pure-numeric tokens (years, page
        ranges, axis ticks) have 0 % alpha but no math chars and are
        therefore preserved — they carry reference-list structure that
        small models need to segment citations.
        """
        if not tok:
            return False
        if sum(c.isalpha() for c in tok) / len(tok) >= 0.40:
            return False  # sufficiently word-like — keep unconditionally
        return any(c in EntityExtractor._MATH_TOKEN_CHARS for c in tok)

    def extract(self, chunk_id: str, text: str) -> ExtractionResult:
        """
        Extract entities and relations from a single text chunk.

        Implements a multi-stage fallback policy:

        1. Strip math (including PDF bracket-subscript tokens).
        2. Skip near-pure-math chunks (<20 word tokens) → ``skipped_math``.
        3. Pre-estimate input tokens; if near context limit, split chunk and
           merge sub-results               → ``predicted_overflow`` for each half.
        4. Call primary model.
        5. Validate output (JSON, schema, truncation).
        6. On format failure: strict repair retry with primary model.
        7. On persistent failure or overflow: re-chunk (if large) or escalate
           to fallback model.
        8. Sparse guard: escalate if dense chunk returns 0 entities.
        9. Normalize entity types and relation labels to canonical sets.

        Returns
        -------
        ExtractionResult
        """
        cleaned = self._strip_math(text)
        tokens = cleaned.split()
        word_tokens = sum(1 for t in tokens if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.60)
        if word_tokens < 20:
            logger.debug("Skipping near-empty chunk %s after math stripping", chunk_id)
            return ExtractionResult(chunk_id=chunk_id, status="skipped_math")

        # Capitalized-token ratio: used by sparse guard in _validate_and_parse.
        cap_ratio = sum(1 for t in tokens if t and t[0].isupper()) / max(len(tokens), 1)

        # -- Step 3: token pre-estimation ---------------------------------------
        if self._provider == LLMProvider.OLLAMA:
            estimated_input = len(tokens) + self._PROMPT_OVERHEAD_TOKENS
            safe_budget = (
                int(self._OLLAMA_NUM_CTX * self._CTX_SAFE_FRACTION) - self._OUTPUT_RESERVE_TOKENS
            )
            if estimated_input > safe_budget and len(tokens) >= 60:
                logger.info(
                    "Chunk %s estimated %d tokens > safe budget %d — splitting.",
                    chunk_id,
                    estimated_input,
                    safe_budget,
                )
                return self._extract_split(chunk_id, cleaned, word_tokens, cap_ratio)

        # -- Steps 4-8: primary call → validate → repair → fallback ------------
        return self._extract_with_policy(chunk_id, cleaned, word_tokens, cap_ratio)

    def _extract_split(
        self, chunk_id: str, text: str, word_tokens: int, cap_ratio: float
    ) -> ExtractionResult:
        """Split *text* into two halves, extract each, and merge."""
        words = text.split()
        mid = len(words) // 2
        halves = [" ".join(words[:mid]), " ".join(words[mid:])]
        merged_entities: list[Entity] = []
        merged_relations: list[Relation] = []
        any_fallback = False
        all_failed = True
        for i, half in enumerate(halves):
            half_id = f"{chunk_id}__split{i}"
            half_toks = half.split()
            half_word_tokens = sum(
                1 for t in half_toks if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.60
            )
            if half_word_tokens < 10:
                continue
            half_cap_ratio = sum(1 for t in half_toks if t and t[0].isupper()) / max(
                len(half_toks), 1
            )
            result = self._extract_with_policy(half_id, half, half_word_tokens, half_cap_ratio)
            if result.status in ("ok", "fallback_success", "predicted_overflow"):
                all_failed = False
            if "fallback" in result.status:
                any_fallback = True
            merged_entities.extend(result.entities)
            # Re-tag relation chunk_id back to parent
            for rel in result.relations:
                rel.chunk_id = chunk_id
            merged_relations.extend(result.relations)
        if all_failed:
            return ExtractionResult(chunk_id=chunk_id, status="fallback_fail")
        final_status = "fallback_success" if any_fallback else "predicted_overflow"
        result = ExtractionResult(
            chunk_id=chunk_id,
            entities=merged_entities,
            relations=merged_relations,
            status=final_status,
        )
        return self._normalize(result)

    def _extract_with_policy(
        self, chunk_id: str, text: str, word_tokens: int, cap_ratio: float
    ) -> ExtractionResult:
        """Run primary model → repair retry → re-chunk → fallback model."""
        # -- Step 4: primary call --
        failure_reason: str | None = None
        raw = self._safe_call_llm(text)
        if raw is None:
            failure_reason = "call_error"
        else:
            result, failure_reason = self._validate_and_parse(chunk_id, raw, word_tokens, cap_ratio)
            if failure_reason is None:
                return self._normalize(result)

        # -- Step 6: repair retry with strict prompt --
        if failure_reason in ("invalid_json", "schema_mismatch", "truncated_output"):
            logger.info(
                "Chunk %s: %s — attempting strict repair retry.",
                chunk_id,
                failure_reason,
            )
            raw2 = self._safe_call_llm(text, repair_mode=True)
            if raw2 is not None:
                result2, failure_reason2 = self._validate_and_parse(
                    chunk_id, raw2, word_tokens, cap_ratio
                )
                if failure_reason2 is None:
                    logger.info("Chunk %s: repair retry succeeded.", chunk_id)
                    return self._normalize(result2)
                failure_reason = failure_reason2

        # -- Step 5/7: re-chunk if input was large and any format error --
        if failure_reason in ("invalid_json", "truncated_output") and len(text.split()) >= 60:
            logger.info("Chunk %s: re-chunking into halves before fallback.", chunk_id)
            words = text.split()
            mid = len(words) // 2
            halves = [" ".join(words[:mid]), " ".join(words[mid:])]
            merged_entities: list[Entity] = []
            merged_relations: list[Relation] = []
            rechunk_ok = False
            for i, half in enumerate(halves):
                half_id = f"{chunk_id}__rechunk{i}"
                half_toks = half.split()
                half_wt = sum(
                    1 for t in half_toks if sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.60
                )
                if half_wt < 10:
                    continue
                half_cap = sum(1 for t in half_toks if t and t[0].isupper()) / max(
                    len(half_toks), 1
                )
                r = self._extract_with_policy(half_id, half, half_wt, half_cap)
                if r.status in ("ok", "fallback_success", "predicted_overflow"):
                    rechunk_ok = True
                merged_entities.extend(r.entities)
                for rel in r.relations:
                    rel.chunk_id = chunk_id
                merged_relations.extend(r.relations)
            if rechunk_ok:
                res = ExtractionResult(
                    chunk_id=chunk_id,
                    entities=merged_entities,
                    relations=merged_relations,
                    status="ok",
                )
                return self._normalize(res)

        # -- Step 7/8: escalate to fallback model --
        if self._fallback_model and self._provider == LLMProvider.OLLAMA:
            logger.warning(
                "Chunk %s: escalating to fallback model %s (reason: %s).",
                chunk_id,
                self._fallback_model,
                failure_reason,
            )
            raw_fb = self._safe_call_llm(text, model_override=self._fallback_model)
            if raw_fb is not None:
                fb_result, fb_failure = self._validate_and_parse(
                    chunk_id, raw_fb, word_tokens, cap_ratio
                )
                if fb_failure is None:
                    fb_result.status = "fallback_success"
                    logger.info("Chunk %s: fallback model succeeded.", chunk_id)
                    return self._normalize(fb_result)
            logger.warning("Chunk %s: fallback model also failed.", chunk_id)
            return ExtractionResult(chunk_id=chunk_id, status="fallback_fail")

        # No fallback configured or not Ollama — log final failure reason.
        # sparse_zero_entity is a policy outcome (model returned empty on a
        # dense chunk but no fallback is available), not a hard error, so
        # log at INFO to avoid flooding the output during full corpus runs.
        status_map = {
            "invalid_json": "invalid_json",
            "truncated_output": "truncated_output",
            "schema_mismatch": "schema_mismatch",
            "sparse_zero_entity": "sparse_zero_entity",
        }
        if failure_reason == "sparse_zero_entity":
            logger.info(
                "Chunk %s: dense chunk returned 0 entities — accepting as empty "
                "(set OLLAMA_FALLBACK_MODEL to enable escalation).",
                chunk_id,
            )
        else:
            logger.warning("Chunk %s: extraction failed — %s.", chunk_id, failure_reason)
        return ExtractionResult(
            chunk_id=chunk_id,
            status=status_map.get(failure_reason, "skipped_llm_error"),
        )

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
                return ExtractionResult(chunk_id=cid, status="failed")

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
            status=data.get("status", "ok"),  # default ok for old checkpoint lines
        )

    # ------------------------------------------------------------------
    # Validation, parsing, and normalization helpers
    # ------------------------------------------------------------------

    def _validate_and_parse(
        self, chunk_id: str, raw: str, word_tokens: int, cap_ratio: float
    ) -> tuple[ExtractionResult, str | None]:
        """Parse *raw* LLM response and validate it.

        Parameters
        ----------
        chunk_id:
            Identifier for the chunk being processed.
        raw:
            The raw string response from the LLM.
        word_tokens:
            Number of alphabetically-dominant tokens in the cleaned chunk.
        cap_ratio:
            Fraction of all tokens in the cleaned chunk that start with an
            uppercase letter.  Used for the dense-chunk sparse guard.

        Returns ``(result, failure_reason)`` where *failure_reason* is
        ``None`` on success or one of the granular status codes on failure.
        """
        # Detect truncation: JSON never closed
        stripped = raw.strip()
        if stripped and not stripped.endswith(("}", "]")):
            logger.debug("Chunk %s: response looks truncated.", chunk_id)
            return ExtractionResult(
                chunk_id=chunk_id, status="truncated_output"
            ), "truncated_output"

        # Try parse (raw, then repaired)
        data = None
        for attempt in (raw, self._repair_json(raw)):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    data = parsed
                    break

        if data is None:
            return ExtractionResult(chunk_id=chunk_id, status="invalid_json"), "invalid_json"

        # Schema check: must have at least one of the expected keys
        if "entities" not in data and "relations" not in data:
            logger.debug("Chunk %s: schema mismatch — missing both keys.", chunk_id)
            return ExtractionResult(chunk_id=chunk_id, status="schema_mismatch"), "schema_mismatch"

        result = self._build_result(chunk_id, data)

        # -- Step 5: sparse guard --
        # A chunk is "dense" when it has many words AND many capitalized terms
        # (domain nouns, proper names, acronyms).  Returning 0 entities on such
        # a chunk is a strong signal of under-extraction.
        is_dense = word_tokens >= self._DENSE_MIN_WORD_TOKENS and cap_ratio >= self._DENSE_CAP_RATIO
        if is_dense and not result.entities:
            logger.info(
                "Chunk %s: dense chunk (%d word tokens, cap_ratio=%.2f) returned 0 entities "
                "— flagging for escalation.",
                chunk_id,
                word_tokens,
                cap_ratio,
            )
            return (
                ExtractionResult(chunk_id=chunk_id, status="sparse_zero_entity"),
                "sparse_zero_entity",
            )

        return result, None

    def _safe_call_llm(
        self,
        text: str,
        repair_mode: bool = False,
        model_override: str | None = None,
    ) -> str | None:
        """Call the LLM and return the raw string, or *None* on error."""
        try:
            return self._call_llm(text, repair_mode=repair_mode, model_override=model_override)
        except Exception:
            logger.exception("LLM call error.")
            return None

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
        5. PDF bracket-subscript tokens: ``word[subscript]`` with non-word
           content in brackets (e.g. ``α[k]``, ``H[kj]``, ``R[k]_req``).
           These are PDF→TXT artefacts that pass the alpha filter but
           confuse the model's JSON-mode grammar.
        6. Math-symbol tokens: non-word tokens (< 40 % alphabetic) that
           contain at least one character from ``_MATH_TOKEN_CHARS``.
           Pure-numeric tokens (years, page ranges, axis ticks) are
           intentionally preserved to maintain citation/reference structure.
        7. Collapse runs of whitespace left by removed segments
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

        # 5 — PDF bracket-subscript tokens: a Greek/Latin letter sequence
        # followed immediately by a bracketed subscript from PDF→TXT flattening.
        # Pattern: one or more word chars, then [non-bracket content].
        # Examples: α[k], H[kj], Q[k], υ[k], R[k]_req, ψ[k].
        # We strip the bracket+subscript part, keeping the root word only
        # when the root is ≥2 alphabetic chars; otherwise remove entirely.
        def _fix_bracket_subscript(m: re.Match) -> str:
            root = m.group(1)
            return root if sum(c.isalpha() for c in root) >= 2 else " "

        text = re.sub(r"(\w+)\[[^\]]{1,20}\]", _fix_bracket_subscript, text)

        # 6 — math-symbol tokens (e.g. "=x^2", "_{k}", "≤0.05")
        # Use in-place regex substitution (not split/join) so that newlines are
        # preserved.  split() collapses \n into spaces, which destroys the
        # line-by-line structure that small LLMs rely on to identify entity
        # boundaries in author lists, figure captions, and reference blocks.
        # See _is_math_token() for the exact filter criterion.
        _is_mt = EntityExtractor._is_math_token
        text = re.sub(r"\S+", lambda m: ("" if _is_mt(m.group()) else m.group()), text)

        # 7 — normalise whitespace: collapse spaces/tabs but keep newlines as
        # structural cues for the LLM.  Strip trailing whitespace per line so
        # that lines emptied by step 6 become true blank lines, then collapse
        # 3+ consecutive blank lines to at most 2.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n", "\n", text)  # trailing spaces before newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
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
    def _call_llm(
        self,
        text: str,
        repair_mode: bool = False,
        model_override: str | None = None,
    ) -> str:
        """Call the configured LLM and return the raw response string.

        Parameters
        ----------
        text:
            The cleaned chunk text to extract from.
        repair_mode:
            When *True*, uses the stricter ``REPAIR_SYSTEM_PROMPT`` and
            ``temperature=0`` with reduced ``num_predict`` to recover from
            format drift.
        model_override:
            If given, use this model instead of ``self.model`` (used for
            fallback escalation).
        """
        if self._provider == LLMProvider.OLLAMA:
            return self._call_ollama(text, repair_mode=repair_mode, model_override=model_override)
        return self._call_openai(text, repair_mode=repair_mode)

    def _call_openai(self, text: str, repair_mode: bool = False) -> str:
        """Call the OpenAI Chat Completions API."""
        from openai import OpenAI

        system = REPAIR_SYSTEM_PROMPT if repair_mode else EXTRACTION_SYSTEM_PROMPT
        client = OpenAI(api_key=self._settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=1024 if repair_mode else 2048,
        )
        return response.choices[0].message.content

    def _call_ollama(
        self,
        text: str,
        repair_mode: bool = False,
        model_override: str | None = None,
    ) -> str:
        """Call a local Ollama server."""
        import requests

        system = REPAIR_SYSTEM_PROMPT if repair_mode else EXTRACTION_SYSTEM_PROMPT
        model = model_override or self.model
        prompt = f"{system}\n\n{text}"
        resp = requests.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Ollama JSON mode: constrains output to valid JSON
                "options": {
                    "num_ctx": self._OLLAMA_NUM_CTX,
                    # Repair mode: tighter budget forces concise output
                    "num_predict": 512 if repair_mode else 1024,
                    "temperature": 0,
                },
            },
            timeout=self._settings.ollama_request_timeout,
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
    def _build_result(chunk_id: str, data: dict) -> ExtractionResult:
        """Build an ExtractionResult from a validated parsed JSON dict.

        Handles small-model quirks:
        - entities/relations as lists of strings instead of dicts
        - Unexpected nesting (e.g. {"entities": {"list": [...]}})
        """
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

        return ExtractionResult(
            chunk_id=chunk_id, entities=entities, relations=relations, status="ok"
        )

    @staticmethod
    def _normalize(result: ExtractionResult) -> ExtractionResult:
        """Canonicalize entity types and relation labels in-place.

        Entity type normalization
        -------------------------
        Maps common model variants to the canonical set:
        Method, Algorithm, Dataset, Metric, Task, Model, Framework,
        Concept, Person, Organisation.

        Relation label normalization
        ----------------------------
        Maps alias relation strings to a canonical relation label so that
        graph edges from the primary model and fallback model stay consistent.
        """
        entity_type_map: dict[str, str] = {
            # Method / Algorithm
            "method": "Method",
            "approach": "Method",
            "technique": "Method",
            "procedure": "Method",
            "algorithm": "Algorithm",
            "algo": "Algorithm",
            # Dataset
            "dataset": "Dataset",
            "data": "Dataset",
            "corpus": "Dataset",
            "benchmark": "Dataset",
            # Metric
            "metric": "Metric",
            "measure": "Metric",
            "evaluation metric": "Metric",
            "score": "Metric",
            # Task
            "task": "Task",
            "problem": "Task",
            "objective": "Task",
            # Model
            "model": "Model",
            "network": "Model",
            "architecture": "Model",
            "neural network": "Model",
            # Framework
            "framework": "Framework",
            "library": "Framework",
            "tool": "Framework",
            "toolkit": "Framework",
            "platform": "Framework",
            "software": "Framework",
            # Concept
            "concept": "Concept",
            "idea": "Concept",
            "theory": "Concept",
            "principle": "Concept",
            # Person
            "person": "Person",
            "author": "Person",
            "researcher": "Person",
            # Organisation
            "organisation": "Organisation",
            "organization": "Organisation",
            "org": "Organisation",
            "institution": "Organisation",
            "company": "Organisation",
            "university": "Organisation",
            "lab": "Organisation",
            # Fallback
            "unknown": "UNKNOWN",
            "": "UNKNOWN",
        }

        relation_map: dict[str, str] = {
            # uses / applies
            "uses": "uses",
            "use": "uses",
            "applies": "uses",
            "apply": "uses",
            "employs": "uses",
            "utilizes": "uses",
            "leverages": "uses",
            # outperforms
            "outperforms": "outperforms",
            "beats": "outperforms",
            "surpasses": "outperforms",
            "exceeds": "outperforms",
            "is better than": "outperforms",
            # extends
            "extends": "extends",
            "builds on": "extends",
            "is based on": "extends",
            "is_based_on": "extends",
            "builds upon": "extends",
            # proposes
            "proposes": "proposes",
            "introduces": "proposes",
            "presents": "proposes",
            "develops": "proposes",
            # trained_on
            "trained_on": "trained_on",
            "trained on": "trained_on",
            "fine-tuned on": "trained_on",
            "finetuned_on": "trained_on",
            # evaluated_on
            "evaluated_on": "evaluated_on",
            "evaluated on": "evaluated_on",
            "tested on": "evaluated_on",
            "benchmarked on": "evaluated_on",
            # part_of
            "part_of": "part_of",
            "part of": "part_of",
            "component of": "part_of",
            "belongs to": "part_of",
            "included in": "part_of",
            # is_variant_of
            "is_variant_of": "is_variant_of",
            "is variant of": "is_variant_of",
            "variant of": "is_variant_of",
            "is a variant of": "is_variant_of",
            # achieves
            "achieves": "achieves",
            "obtains": "achieves",
            "reports": "achieves",
            # requires
            "requires": "requires",
            "depends on": "requires",
            "needs": "requires",
        }

        known_types = {
            "Method",
            "Algorithm",
            "Dataset",
            "Metric",
            "Task",
            "Model",
            "Framework",
            "Concept",
            "Person",
            "Organisation",
            "UNKNOWN",
        }

        for entity in result.entities:
            if entity.entity_type not in known_types:
                normalized = entity_type_map.get(entity.entity_type.lower().strip(), None)
                entity.entity_type = normalized if normalized else "UNKNOWN"
            # Title-case entity names
            if entity.name:
                entity.name = entity.name.strip()

        for relation in result.relations:
            normalized_rel = relation_map.get(relation.relation_type.lower().strip(), None)
            if normalized_rel:
                relation.relation_type = normalized_rel
            else:
                # Keep unknown relations but lowercase + underscored
                relation.relation_type = relation.relation_type.lower().strip().replace(" ", "_")

        return result
