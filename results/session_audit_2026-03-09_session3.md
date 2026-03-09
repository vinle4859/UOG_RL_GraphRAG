# Session Audit Report — Session 3
**Date:** 2026-03-09
**Branch:** `feat/resilience-preprocessing-retry`
**Authored by:** GitHub Copilot (Claude Sonnet 4.6)

---

## 1. Summary

This session diagnosed and fixed a series of cascading failures that prevented
the `build-graph` pipeline from producing any output.  Root causes were:

1. `qwen3.5:0.8b` (a Qwen3 "thinking" model) ignores `think: false` and exhausts
   `num_predict` tokens on internal reasoning, returning an empty `response` field.
2. `llama3.2:3b` (the fallback) prepends prose before the JSON and hits the
   `num_predict: 512` cap mid-object, leaving JSON unclosed.
3. Three parallel Ollama workers caused request queuing that exceeded the 90 s
   HTTP timeout.

The final resolution was switching the active model to `gemma3:1b` (815 MB, fits
entirely in VRAM) with `format: json` constrained decoding, a single worker, and
a 180 s timeout.  A 2-document validation build confirmed **86 nodes / 42 edges**
extracted from 43 chunks with only 1 skip (~4 s/chunk).

Additionally, a latent refactor centralised all Ollama HTTP timeouts into a single
`ollama_request_timeout` settings field.

---

## 2. Configuration Changes (`.env` — gitignored)

| Variable | Before | After | Reason |
|----------|--------|-------|--------|
| `LLM_MODEL_NAME` | `qwen3.5:0.8b` | `gemma3:1b` | qwen3.5 always thinks, empties response |
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | `gemma3:1b` | same |

---

## 3. Commits This Session

### Commit `cee3023` — `fix: switch to llama3.2:3b; qwen3.5:0.8b ignores think:false`

Root-cause diagnosis of the `qwen3.5:0.8b` empty-response bug.

| File | Change |
|------|--------|
| `src/graph_rag/entity_extractor.py` | Reverted model to `llama3.2:3b`; added `<think>…</think>` stripper in `_repair_json()` as forward-compat safety net; bumped `num_ctx` to 2048 |

**Diagnosis path:**
- Raw Ollama probe: `response: ''` with `done_reason: length` and `eval_count: 512`
- The model's `thinking` field contained 6000+ chars — entire budget consumed by reasoning chain
- `think: False` option and `/no_think` prompt suffix both ignored by `qwen3.5:0.8b`

---

### Commit `9a38ca5` — `fix: add format:json and bump num_predict to 1024 for llama3.2:3b`

Fixes `llama3.2:3b` truncation: model prepended "Here is the JSON..." prose,
hitting the 512-token cap mid-object.

| File | Change |
|------|--------|
| `src/graph_rag/entity_extractor.py` | Added `"format": "json"` to Ollama request body; raised `num_predict` 512 → 1024 |

**Result:** All 3 test chunks returned `done_reason: stop` with valid, parseable JSON.

---

### Commit `1a2a320` — `fix: set Ollama to 1 worker and 180s timeout to prevent queue timeouts`

With `format: json` constrained decoding taking ~30 s/chunk on the 3B model,
three parallel workers caused the third queued request to wait `3 × 30 s = 90 s`
before even starting — exceeding the timeout.

| File | Change |
|------|--------|
| `src/graph_rag/entity_extractor.py` | `_DEFAULT_OLLAMA_WORKERS` 3 → 1; `timeout` 90 → 180 s |

**Note:** Ollama queues GPU requests internally — parallelism only adds latency
for a single-GPU setup.

---

## 4. Model Selection — Final Decision

| Model | Size | Speed | Success rate | Notes |
|-------|------|-------|-------------|-------|
| `qwen3.5:0.8b` | 1.0 GB | — | 0 % | Always generates `<think>` blocks; ignores `think:false` |
| `llama3.2:3b` | 2.0 GB | ~30 s/chunk | ~100 % JSON valid | Requires `format:json`; still slow due to VRAM pressure |
| `gemma3:1b` ✅ | 815 MB | **~4 s/chunk** | **42/43 (98 %)** | Chosen: smallest, fastest, fits VRAM, reliable JSON |

Active model switched in `.env`:
```
LLM_MODEL_NAME=gemma3:1b
OLLAMA_MODEL=gemma3:1b
```

---

## 5. Code Changes Pending Commit (Session 3 — uncommitted)

Files modified but not yet staged — clean refactor to centralise Ollama timeouts:

| File | Change |
|------|--------|
| `src/config.py` | Added `ollama_request_timeout: int = 300` field to `Settings` |
| `src/evaluation/metrics.py` | `timeout=120` → `timeout=settings.ollama_request_timeout` |
| `src/evaluation/questions.py` | `timeout=180` → `timeout=settings.ollama_request_timeout` |
| `src/graph_rag/community.py` | `timeout=180` → `timeout=settings.ollama_request_timeout` |
| `src/rag/generator.py` | `timeout=120` → `timeout=settings.ollama_request_timeout` |

---

## 6. Validation Build Results

**Command:** `rag-bench build-graph --limit 2`
**Model:** `gemma3:1b` with `format:json`

| Metric | Value |
|--------|-------|
| Documents processed | 2 (`1701.01952`, `1701.02710`) |
| Chunks processed | 43 |
| Chunks skipped | 1 (math-heavy section) |
| Extraction time | 2 min 50 s (~4 s/chunk) |
| Graph nodes | 86 |
| Graph edges | 42 |
| Communities (Louvain) | 56 |
| Graph file size | 22 KB |

---

## 7. Next Steps

- [ ] Run full corpus build: `rag-bench build-graph --checkpoint data/graphs/extraction_checkpoint.jsonl`
- [ ] Run benchmark: `rag-bench benchmark data/eval_questions.json --graphrag-mode local --output results/benchmark_full.csv`
- [ ] Consider `pip install leidenalg python-igraph` for better community detection than Louvain
