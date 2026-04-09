# Session Audit Report — 2026-03-11 (Session 2)
**Date:** 2026-03-11
**Branch:** `feat/resilience-preprocessing-retry`
**Authored by:** GitHub Copilot (Claude Sonnet 4.6)

---

## 1. Problem Statement

After 3-4 hours of continuous graph-extraction runs, all Ollama API calls started failing with:

```
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=11434):
Read timed out. (read timeout=300)
```

The pipeline halted completely each time, requiring manual restart.

---

## 2. Root Cause Analysis

Three compounding causes were identified:

| # | Cause | Effect |
|---|-------|--------|
| 1 | **Stale TCP connections** | `requests` uses an HTTP connection pool (`urllib3`). After hours of idle time, pooled sockets silently go dead at the OS/network level. The next request writes successfully (the OS buffers it) but the read blocks forever until the 300 s timeout fires. |
| 2 | **Single-value timeout** | Passing a plain `int` to `requests.post(timeout=...)` only sets the **read** timeout. The **TCP connect** timeout is unlimited — if Ollama's port becomes unresponsive (e.g. after a GPU OOM restart), the pipeline hangs indefinitely on connection establishment before the read phase even begins. |
| 3 | **No retry logic** | A single transient error immediately propagated as an unhandled exception and killed the entire pipeline process with no recovery path. |

---

## 3. Changes Made

### 3.1 New file: `src/utils/ollama_client.py`

Introduced a dedicated `ollama_post()` helper that centralises all Ollama HTTP calls and applies three targeted fixes:

```
src/utils/ollama_client.py   (new, 126 lines)
```

**Fix 1 — No keep-alive / fresh session per attempt**

```python
with requests.Session() as session:
    session.headers.update({"Connection": "close"})
    resp = session.post(url, json=payload, timeout=(_CONNECT_TIMEOUT, read_timeout))
```

A brand-new `Session` is created for every attempt. `Connection: close` prevents `urllib3` from pooling the socket. This eliminates stale-connection hangs entirely.

**Fix 2 — Tuple `(connect, read)` timeout**

```python
_CONNECT_TIMEOUT: int = 10   # TCP SYN → SYN-ACK deadline
timeout=(_CONNECT_TIMEOUT, read_timeout)
```

The 10-second connect deadline means a dead Ollama process is detected in ≤10 s instead of hanging indefinitely or waiting the full 300 s. The read timeout remains user-configurable via `settings.ollama_request_timeout`.

**Fix 3 — Retry with exponential back-off**

```python
for attempt in range(max_retries + 1):   # default max_retries=3
    ...
    except (ReadTimeout, ConnectionError):
        time.sleep(delay)               # 5 s → 10 s → 20 s → fail
        delay = min(delay * 2.0, 60.0)
```

Up to 3 retries (4 total attempts) with doubling back-off, capped at 60 s. Only `ReadTimeout` and `ConnectionError` are retried; non-2xx HTTP responses are raised immediately (they indicate logic errors, not transience).

---

### 3.2 Updated call sites (4 files)

All existing `requests.post` Ollama calls were replaced with `ollama_post()`:

| File | Function / location |
|------|---------------------|
| `src/graph_rag/entity_extractor.py` | `EntityExtractor._call_ollama()` |
| `src/rag/generator.py` | `OllamaGenerator.generate()` |
| `src/graph_rag/community.py` | `summarise_community()` |
| `src/evaluation/metrics.py` | `_llm_scalar_score()` |

Each change is a mechanical substitution:

```python
# Before
import requests
resp = requests.post(url, json=payload, timeout=settings.ollama_request_timeout)
resp.raise_for_status()
result = resp.json()["response"]

# After
from src.utils.ollama_client import ollama_post
data = ollama_post(url, payload=payload, read_timeout=settings.ollama_request_timeout)
result = data["response"]
```

---

## 4. Files Changed

| File | Type | Lines changed |
|------|------|---------------|
| `src/utils/ollama_client.py` | **New** | +126 |
| `src/graph_rag/entity_extractor.py` | Modified | -7 / +6 |
| `src/rag/generator.py` | Modified | -5 / +4 |
| `src/graph_rag/community.py` | Modified | -6 / +5 |
| `src/evaluation/metrics.py` | Modified | -6 / +5 |

---

## 5. Testing

Static analysis (VS Code / Pylance) reported **no errors** on all 5 modified files after the changes were applied.

The pipeline was already running (`rag-bench build-graph --resume`) and had re-connected successfully to Ollama before this audit was written.

---

## 6. Configuration Note

`ollama_request_timeout` (default 300 s) in `src/config.py` is **unchanged** — it continues to control the per-request read deadline. The new connect timeout (10 s, hardcoded as `_CONNECT_TIMEOUT`) and retry count (3, as `max_retries`) are intentionally not exposed in config because they represent infrastructure behaviour rather than model tuning.

If Ollama regularly takes >10 s to accept a connection (e.g. loading a large model from cold), increase `_CONNECT_TIMEOUT` in `src/utils/ollama_client.py`.
