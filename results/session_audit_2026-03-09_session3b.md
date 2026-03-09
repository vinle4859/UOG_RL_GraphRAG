# Session Audit Report — Session 3 (Extended)
**Date:** 2026-03-09
**Branch:** `feat/resilience-preprocessing-retry`
**Authored by:** GitHub Copilot (Claude Sonnet 4.6)

---

## 1. Summary of All Changes This Session

This is the comprehensive audit of all commits made since `origin/feat/resilience-preprocessing-retry`
(i.e. since `14ad7a8`).  The session spanned model selection, math filtering, performance tuning,
and extraction quality diagnostics.

---

## 2. Commit Log (This Session — 9 Commits)

| # | SHA | Message |
|---|-----|---------|
| 1 | `0bef5a2` | feat: resilience + observability improvements *(catchup from session 1)* |
| 2 | `545216d` | feat(extractor): skip math-heavy chunks before LLM call |
| 3 | `233881f` | feat(extractor): strip math expressions instead of skipping whole chunk |
| 4 | `14252ec` | docs: session 2 audit report |
| 5 | `7ba6e7d` | perf: parallel Ollama extraction + reduced context window |
| 6 | `cee3023` | fix: switch to llama3.2:3b; qwen3.5:0.8b exhausts token budget on reasoning |
| 7 | `9a38ca5` | fix: add format:json and bump num_predict to 1024 for llama3.2:3b |
| 8 | `1a2a320` | fix: set Ollama to 1 worker and 180s timeout to prevent queue timeouts |
| 9 | `4f00f0a` | refactor: centralise Ollama timeout into settings + session 3 audit |
| 10 | `650bf61` | feat: add status field to ExtractionResult checkpoint |

---

## 3. File-Level Changes

### `src/graph_rag/entity_extractor.py`

| Change | Commits |
|--------|---------|
| `ExtractionResult` gains `status: str` field (`"ok"` / `"skipped_math"` / `"skipped_llm_error"` / `"failed"`) | `650bf61` |
| `_repair_json()` strips `<think>…</think>` Qwen3 reasoning blocks | `cee3023` |
| `extract()` calls `_strip_math()` first; skips if <20 word tokens (`status="skipped_math"`) | `233881f`, `650bf61` |
| `_strip_math()` — 6-step regex pipeline (LaTeX envs, display math, inline math, bare cmds, stray symbols, whitespace) | `233881f` |
| `_is_math_heavy()` — 3-layer diagnostic helper retained | `545216d` |
| `extract_batch()` — `ThreadPoolExecutor` with `_DEFAULT_OLLAMA_WORKERS=1` (sequential GPU) | `7ba6e7d`, `1a2a320` |
| `_call_ollama()` — `format:"json"`, `num_ctx:2048`, `num_predict:1024`, `timeout:180s` | `9a38ca5`, `1a2a320` |
| `_deserialize()` — reads `status` with `"ok"` default for backward-compat old checkpoints | `650bf61` |
| `_extract_one()` — returns `status="failed"` on exception | `650bf61` |
| JSONL checkpoint writes include `status` field | `650bf61` |

### `src/config.py`
- Added `ollama_request_timeout: int = 300` settings field (`4f00f0a`)

### `src/rag/generator.py`, `src/graph_rag/community.py`, `src/evaluation/metrics.py`, `src/evaluation/questions.py`
- All hard-coded Ollama `timeout=` values replaced with `settings.ollama_request_timeout` (`4f00f0a`)

### `src/cli.py`
- Added `--limit`, `--resume`, `--checkpoint` flags to `build-graph` command (`0bef5a2`)

### `src/graph_rag/pipeline.py`
- Incremental indexing with doc manifest, deduplication, `tqdm` progress bars (`0bef5a2`)

### `src/utils/logging_setup.py`
- Rotating file handler; auto-dated log path `results/logs/graphrag_YYYYMMDD.log` (`0bef5a2`)

### `src/data/chunker.py`
- Unicode `→` replaced with `->` in 3 logger calls (Windows cp1252 fix) (`7ba6e7d`)

### `.env` (gitignored — not committed)
```
LLM_MODEL_NAME=gemma3:1b
OLLAMA_MODEL=gemma3:1b
```
Active model switched from `qwen3.5:0.8b` → `llama3.2:3b` → `gemma3:1b`.
`gemma3:1b` (815 MB) chosen for speed (~4 s/chunk) and reliable JSON output.

---

## 4. Checkpoint Schema

Every processed chunk is now recorded in `data/graphs/extraction_checkpoint.jsonl`:

```jsonl
{"chunk_id": "1701.01952__chunk_4",  "entities": [...], "relations": [...], "status": "ok"}
{"chunk_id": "1701.01952__chunk_18", "entities": [],     "relations": [],     "status": "skipped_math"}
{"chunk_id": "1701.01952__chunk_31", "entities": [],     "relations": [],     "status": "skipped_llm_error"}
```

**Filtering failed chunks for rerun:**
```powershell
Get-Content data/graphs/extraction_checkpoint.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.status -eq "skipped_llm_error" -or $_.status -eq "failed" } |
  Select-Object chunk_id, status |
  Format-Table
```

---

## 5. Validation Build (2 docs, `gemma3:1b`)

| Metric | Value |
|--------|-------|
| Chunks | 43 |
| Skipped | 1 (`skipped_math`) |
| Speed | ~4 s/chunk |
| Nodes | 86 |
| Edges | 42 |
| Communities | 56 (Louvain) |

---

## 6. Full Corpus Rerun Commands

See section below — also provided directly to the user.
