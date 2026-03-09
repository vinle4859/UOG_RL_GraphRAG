# Session Audit Report
**Date:** 2026-03-09
**Branch:** `feat/resilience-preprocessing-retry`
**Authored by:** GitHub Copilot (Claude Sonnet 4.6)

---

## 1. Summary

This session diagnosed and fixed a cascading set of failures blocking `rag-bench build-graph`, then extended the pipeline with build reliability and observability features. All non-experimental changes were committed and pushed to the remote branch.

---

## 2. Bugs Fixed

### 2.1 Ollama model mismatch — 404 on `/api/generate`
| | |
|---|---|
| **File** | `.env` |
| **Symptom** | `requests.exceptions.HTTPError: 404 Not Found` on `http://localhost:11434/api/generate` |
| **Root cause** | `EntityExtractor` reads `settings.ollama_model` (defaulting to `qwen2.5:7b`), while `.env` only set `LLM_MODEL_NAME=llama3.2:3b`. The two fields are separate. Ollama returned 404 because `qwen2.5:7b` was not installed. |
| **Fix** | Added `OLLAMA_MODEL=llama3.2:3b` to `.env` so both fields point to the installed model. |

---

### 2.2 `AttributeError: 'str' object has no attribute 'get'` in entity extractor
| | |
|---|---|
| **File** | `src/graph_rag/entity_extractor.py` |
| **Commit** | `14ad7a8` |
| **Symptom** | Crash on `e.get("name", "")` during extraction of certain chunks. |
| **Root cause** | `llama3.2:3b` occasionally returns `"entities": ["PPO", "RL"]` — a list of plain strings instead of dicts. The original code assumed every item was a dict. |
| **Fix** | `_parse_response` now checks `isinstance(e, dict)` before calling `.get()`. Plain strings are wrapped into minimal `Entity(name=..., entity_type="UNKNOWN")` objects. |

**Additional hardening in same commit:**
- `_repair_json`: unwraps array-wrapped responses (`[{...}]` → `{...}`)
- `_repair_json`: validates returned JSON is a `dict` (not a list or scalar)
- `_parse_response`: accepts both `"relation"` and `"relation_type"` key names
- `EXTRACTION_SYSTEM_PROMPT`: added explicit instruction to return `{"entities": [], "relations": []}` for math-heavy / unextractable chunks

---

## 3. Features Added

### 3.1 ChunkingPipeline — preprocessing stage before tokenization
| | |
|---|---|
| **File** | `src/data/chunker.py` |
| **What** | New `ChunkingPipeline` class wrapping configurable text-cleaning steps before `TokenChunker`. |

**Preprocessing steps (applied in order):**

| Step | Function | Effect |
|------|----------|--------|
| 1 | `_remove_control_characters` | Strips non-printable bytes (preserves `\n`, `\t`) |
| 2 | `_normalize_unicode` | NFC normalisation — combining chars → single code-points |
| 3 | `_collapse_whitespace` | Collapses 3+ blank lines → 1; multiple spaces → 1 |

Both `RAGPipeline` and `GraphRAGPipeline` were updated to use `ChunkingPipeline` instead of `TokenChunker` directly.

---

### 3.2 Build Reliability
| | |
|---|---|
| **Files** | `src/graph_rag/entity_extractor.py`, `src/graph_rag/pipeline.py` |

#### Checkpointing / Resume (`entity_extractor.py`)
- `extract_batch()` accepts a `checkpoint_path` (JSONL file)
- After each chunk, the `ExtractionResult` is serialized and appended; file is flushed immediately (crash-safe)
- On re-run with the same `checkpoint_path`, already-completed `chunk_id`s are loaded and skipped
- New `_deserialize()` static method reconstructs `ExtractionResult` from stored JSON

#### Incremental Indexing (`pipeline.py`)
- `build()` now accepts `limit`, `resume`, and `checkpoint_path` parameters
- A manifest file (`data/graphs/indexed_docs.json`) records fully indexed `doc_id`s
- `--resume`: filters out already-indexed docs before chunking/extraction; reads extraction checkpoint
- `--limit N`: caps the number of documents processed (for smoke-testing)

#### Document Deduplication (`pipeline.py`)
- Duplicate `doc_id`s are eliminated before any processing (first occurrence wins)

---

### 3.3 Observability

#### Progress Bars (`entity_extractor.py`, `pipeline.py`)
- `tqdm` bar on entity extraction: `Extracting entities: 42/61458 [chunk]`
- `tqdm` bar on vector store indexing: `Indexing chunks: 5/960 [batch]`
- `tqdm` bar on community summarisation: `Summarising communities: 3/47 [community]`

#### Log-to-file (`src/utils/logging_setup.py`)
- `setup_logging()` now accepts an optional `log_file` parameter
- Default: auto-creates `results/logs/graphrag_YYYYMMDD.log` on every run
- Uses `RotatingFileHandler`: 10 MB max per file, 5 backups retained
- Pass `log_file=False` to disable file logging; pass a custom `Path` to redirect

---

### 3.4 CLI Extensions (`src/cli.py`)

New flags added to `rag-bench build-graph`:

| Flag | Type | Description |
|------|------|-------------|
| `--limit N` | int | Process only the first N documents |
| `--resume` | flag | Skip already-indexed docs; load extraction checkpoint |
| `--checkpoint PATH` | path | Explicit JSONL checkpoint file location |
| `--no-summarise` | flag | Skip community summarisation *(pre-existing)* |

**Usage examples:**
```bash
# Test on 10 documents only
rag-bench build-graph --limit 10

# Resume a crashed run
rag-bench build-graph --resume

# Test resume on a small subset
rag-bench build-graph --limit 50 --resume

# Custom checkpoint file
rag-bench build-graph --resume --checkpoint data/graphs/my_checkpoint.jsonl

# Skip slow community summarisation
rag-bench build-graph --no-summarise
```

---

## 4. Diagnosis: Why Math-Heavy Chunks Produce Invalid JSON

Chunk `1701.01952__chunk_11` (and similar) fail entity extraction due to:

1. **Square brackets in notation** (`H[kj]`, `Q[k]`, `v[j]`) — the model mistakes them for JSON array syntax mid-generation
2. **Unicode math symbols** (`†`, `≤`, `∑`, `ζ`, `λ`) — poor tokenization causes output corruption in small models
3. **No natural language** — purely proof-based content; the 3B model produces narrative text instead of JSON when it has nothing to extract
4. **PDF-to-text artefacts** — subscripts/superscripts flattened into unreadable token sequences (`ζPt PK j=1 r λmax`)

**Recommended mitigations (not yet implemented):**
- Switch to `qwen2.5:7b` (`ollama pull qwen2.5:7b` + `OLLAMA_MODEL=qwen2.5:7b` in `.env`) — handles structured output over math much better
- Section-aware chunking — split at headings first so proofs stay isolated from text sections
- Unicode normalization before LLM calls (replace `†` → `"conjugate transpose"`, etc.)

---

## 5. Files Changed This Session

| File | Change type |
|------|------------|
| `.env` | Fix — added `OLLAMA_MODEL=llama3.2:3b` |
| `src/data/chunker.py` | Feature — `ChunkingPipeline`, preprocessing functions |
| `src/rag/pipeline.py` | Update — use `ChunkingPipeline` |
| `src/graph_rag/pipeline.py` | Feature — `limit`, `resume`, `checkpoint_path`, `tqdm`, manifest, dedup |
| `src/graph_rag/entity_extractor.py` | Fix + Feature — crash fix, checkpointing, `tqdm`, prompt hardening |
| `src/utils/logging_setup.py` | Feature — file logging with rotation |
| `src/cli.py` | Feature — `--limit`, `--resume`, `--checkpoint` flags |

---

## 6. Git Commits

| Hash | Message |
|------|---------|
| `14ad7a8` | fix: handle malformed LLM JSON in entity extractor for small models |
| `6b3cecd` | test: add smoke test / usage example for RAGPipeline *(pre-session baseline)* |

> **Note:** Build reliability and observability changes (sections 3.2–3.4) were made after the last push and are pending a follow-up commit.

---

## 7. Pending Actions

- [ ] Commit and push the build reliability + observability changes
- [ ] Pull `qwen2.5:7b` and set `OLLAMA_MODEL=qwen2.5:7b` to reduce extraction failures
- [ ] Create `data/eval_questions.json` to unblock `rag-bench benchmark`
- [ ] Re-run `rag-bench build-graph --limit 20` to validate resume + checkpoint behaviour
