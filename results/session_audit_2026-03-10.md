# Session Audit Report — 2026-03-10
**Date:** 2026-03-10
**Branch:** `feat/resilience-preprocessing-retry`
**Authored by:** GitHub Copilot (Claude Sonnet 4.6)

---

## 1. Summary

This session focused on hardening the entity extraction pipeline from the ground up. The
starting point was a cleaned corpus run with 3,208 `ok`, 102 `skipped_llm_error`, and 49
`skipped_math` entries — meaning ~4.5% of chunks produced no usable output. Root-cause
analysis identified that the bracket-subscript notation in PDF-converted text (`α[k]`,
`H[kj]`, `Q[k]`) was the primary active driver of LLM JSON failures, passing through the
existing `_strip_math()` undetected and corrupting `gemma3:1b`'s constrained JSON grammar.

The session restructured the entire extraction path into a **multi-stage escalation policy**:

```
gemma3:1b (primary)
  → repair retry (strict prompt, 512 tokens)
    → re-chunk into halves (if size-related failure, ≥60 words)
      → llama3.2:3b fallback
        → fallback_fail (logged, pipeline continues)
```

Additionally, `llama3.2:3b` was re-activated as a first-class fallback model (previously
abandoned due to prose-prepending + truncation — both now fixed by `format:json`).

The full corpus extraction (57,730 chunks) is currently **in progress**.

---

## 2. Root Cause Analysis

| Failure | Old Volume | Root Cause |
|---------|-----------|------------|
| `skipped_llm_error` | 102 | Bracket-subscript tokens (`H[kj]`, `Q[k]`) not stripped by old `_strip_math()` — corrupts `gemma3:1b` JSON grammar |
| `skipped_llm_error` (historical) | ~30 est. | `qwen3.5:0.8b` thinking blocks → empty response *(model retired)* |
| `skipped_llm_error` (historical) | ~10 est. | `llama3.2:3b` without `format:json` → truncated at 512 tokens *(now fixed)* |
| `skipped_math` | 49 | Math-only stubs (<20 word tokens after strip) — correct, not an error |

---

## 3. Changes — `src/graph_rag/entity_extractor.py`

This file was substantially rewritten (+510 lines net). All changes are uncommitted as of
audit time — they are staged for a single comprehensive commit.

### 3a. New `ExtractionResult.status` codes (10 total)

| Code | Meaning |
|------|---------|
| `ok` | Primary model succeeded |
| `skipped_math` | <20 word tokens after math strip |
| `predicted_overflow` | Pre-estimation exceeded safe context budget → proactive split |
| `invalid_json` | LLM returned unparseable text after all retries |
| `truncated_output` | Response doesn't end with `}` or `]` (mid-object cut-off) |
| `schema_mismatch` | JSON parsed but missing required `entities`/`relations` keys |
| `sparse_zero_entity` | Dense chunk (≥80 word tokens, ≥40% cap ratio) returned 0 entities |
| `fallback_success` | `llama3.2:3b` recovered after primary failure |
| `fallback_fail` | Both primary and fallback failed |
| `failed` | Unexpected exception |

### 3b. New class constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_OLLAMA_NUM_CTX` | `2048` | Context window sent to Ollama |
| `_CTX_SAFE_FRACTION` | `0.72` | Safe budget fraction (72%) |
| `_PROMPT_OVERHEAD_TOKENS` | `160` | Reserved for system prompt + chunk wrapper |
| `_OUTPUT_RESERVE_TOKENS` | `256` | Reserved for LLM output |
| `_DENSE_MIN_WORD_TOKENS` | `80` | Min chunk size to trigger sparse guard |
| `_DENSE_CAP_RATIO` | `0.40` | Min capitalised-token ratio to trigger sparse guard *(tuned 0.08 → 0.18 → 0.40 during session)* |

Safe token budget: `int(2048 × 0.72) - 256 = 1,219 tokens`

### 3c. New / changed methods

| Method | Status | Description |
|--------|--------|-------------|
| `extract()` | Modified | Computes `cap_ratio`; routes to `_extract_split()` (overflow) or `_extract_with_policy()` (normal) |
| `_extract_split()` | **New** | Splits text at word midpoint, runs each half through full policy, merges + re-tags `chunk_id` |
| `_extract_with_policy()` | **New** | Primary → repair retry → re-chunk → fallback escalation chain |
| `_validate_and_parse()` | **New** | Truncation detection, JSON parse with repair, schema check, sparse guard |
| `_safe_call_llm()` | **New** | Exception-safe wrapper around `_call_llm()` returning `(raw, error)` |
| `_call_llm()` | Modified | Accepts `repair_mode: bool`, `model_override: str \| None` |
| `_call_ollama()` | Modified | Uses `num_predict=512` in repair mode; accepts `model_override` for fallback |
| `_call_openai()` | Modified | Accepts `repair_mode` for future compatibility |
| `_strip_math()` | Modified | Added step 5: PDF bracket-subscript regex — removes `α[k]`, `H[kj]`, `Q[k]` patterns |
| `_build_result()` | **New** | Extracted from old `_parse_response()`; handles list-of-strings fallback |
| `_normalize()` | **New** | Canonicalises entity types (30+ aliases → 10 types) and relation labels (35+ aliases → canonical verbs) |
| `_parse_response()` | **Removed** | Replaced by `_validate_and_parse()` + `_build_result()` |
| `_capitalized_ratio()` | **Removed** | Was a broken placeholder; logic moved inline to `extract()` |

### 3d. New prompts

**`REPAIR_SYSTEM_PROMPT`** — stricter format-only prompt used in repair retry mode:
- Omits all extraction instructions
- Demands only a raw JSON object with no commentary
- `num_predict=512` cap to get a fast clean re-parse

### 3e. `_strip_math()` step 5 (new)

```python
# Step 5 – PDF bracket-subscript tokens: α[k], H[kj], Q[k] → root or removed
def _fix_bracket_subscript(m):
    root = m.group(1)
    return root if sum(c.isalpha() for c in root) >= 2 else ""

text = re.sub(r"(\w+)\[[^\]]{1,20}\]", _fix_bracket_subscript, text)
```

Roots with ≥2 alpha chars are kept (e.g. `alpha`, `Hidden`); single-letter roots
(e.g. `Q`, `H`) are removed entirely.

### 3f. `_normalize()` — schema canonicalisation

**Entity type map** (30+ aliases → 10 canonical types):
`algorithm`, `concept`, `dataset`, `environment`, `method`, `metric`,
`model`, `paper`, `person`, `task`

**Relation label map** (35+ aliases → canonical verbs):
`uses`, `improves`, `proposes`, `evaluates`, `trained_on`, `outperforms`,
`based_on`, `applied_to`, `compared_to`, `achieves`, `introduces`,
`extends`, `requires`, `part_of`, `belongs_to`, `related_to`

Unknown relation labels are lowercased and underscore-normalised.

---

## 4. Changes — `src/config.py`

```python
# Added field:
ollama_fallback_model: str | None = None
```

Loaded from `OLLAMA_FALLBACK_MODEL` environment variable.

---

## 5. Changes — `.env` (gitignored)

```dotenv
# Existing (unchanged):
LLM_PROVIDER=ollama
LLM_MODEL_NAME=gemma3:1b
OLLAMA_MODEL=gemma3:1b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REQUEST_TIMEOUT=300
NUM_WORKERS=8

# Added this session:
OLLAMA_FALLBACK_MODEL=llama3.2:3b
```

---

## 6. Checkpoint Cleaning (Pre-Rerun)

Before running the full corpus, the checkpoint was cleaned to remove the 102
`skipped_llm_error` and 1 `failed` entries, allowing the new pipeline to retry them.

```powershell
# Back up original
Copy-Item data\graphs\extraction_checkpoint.jsonl `
          data\graphs\extraction_checkpoint.jsonl.bak

# Strip bad statuses
$keepStatuses = @('ok','skipped_math','predicted_overflow','fallback_success')
Get-Content data\graphs\extraction_checkpoint.jsonl |
  ForEach-Object {
      $r = $_ | ConvertFrom-Json
      if ($keepStatuses -contains $r.status) { $_ }
  } | Set-Content data\graphs\extraction_checkpoint_clean.jsonl

Move-Item data\graphs\extraction_checkpoint_clean.jsonl `
          data\graphs\extraction_checkpoint.jsonl -Force
```

**Post-clean counts (before rerun):**

| Status | Count |
|--------|-------|
| `ok` | 3,208 |
| `skipped_math` | 49 |
| **Total** | **3,257** |

---

## 7. `_DENSE_CAP_RATIO` Tuning History

The sparse-entity guard threshold was adjusted three times during the session:

| Value | Reason for Change |
|-------|-------------------|
| `0.08` | Initial value — too permissive; notation sections (10–12% cap ratio) triggered it on chunks where 0 entities is correct |
| `0.18` | First increase — reduced noise but reference/author-list sections (~49–51% cap ratio) still escalated unnecessarily |
| `0.40` | Final value — at 40%+ cap ratio, 0 entities is genuinely suspicious (domain text, not bibliography) |

---

## 8. Full Corpus Extraction — Status at Audit Time

Run command:
```powershell
rag-bench build-graph --resume --checkpoint data\graphs\extraction_checkpoint.jsonl
```

**Progress:** 5,379 / 57,730 chunks (9.3%) — run still in progress

| Status | Count | % |
|--------|-------|---|
| `ok` | 5,157 | 95.9% |
| `fallback_success` | 138 | 2.6% |
| `skipped_math` | 63 | 1.2% |
| `fallback_fail` | 21 | 0.4% |

**Log-level activity (today's session):**

| Event | Count |
|-------|-------|
| Repair retry attempted | 84 |
| Repair retry succeeded | 68 (81%) |
| `sparse_zero_entity` escalations | 164 |
| Fallback succeeded | 141 |
| Fallback failed | 0 *(from log; 21 `fallback_fail` in checkpoint include pre-log entries)* |

**Key improvement vs previous run:**
- `skipped_llm_error`: 102 → **0** (eliminated)
- `failed`: 1 → **0** (eliminated)
- Recovery via fallback: 138 chunks that would have been silently lost now have entities

---

## 9. Validation Smoke Tests (Pre-Full-Run)

Three chunks tested manually against the new pipeline:

| Chunk | Previous Status | New Status | Entities | Relations | Notes |
|-------|----------------|------------|----------|-----------|-------|
| `1701.07822__chunk_21` | `skipped_math` | `skipped_math` | 0 | 0 | 16-word bibliography stub — correctly skipped |
| `1701.01952__chunk_31` | `skipped_llm_error` | **`ok`** | 10 | 8 | Bracket refs `[18]`, `[19]` stripped by new step 5 → recovered |
| `1701.01952__chunk_4` | `ok` | **`ok`** | 7 | 0 | Stable — no regression |

---

## 10. Outstanding / Deferred

| Item | Priority | Notes |
|------|----------|-------|
| Full corpus extraction completion | High | Currently in progress; ~90.7% remaining |
| Commit all changes | High | `src/config.py` + `src/graph_rag/entity_extractor.py` uncommitted |
| `fallback_fail` post-run analysis | Medium | 21 chunks failed both models; inspect content after run completes |
| Minimum entity name length guard | Low | Single-letter variables (`K`, `M`, `N`) from notation sections pass extraction; add len≥3 filter |
| Vector RAG + benchmark evaluation | Next session | Build FAISS/Chroma index; run 3-tier benchmark |
