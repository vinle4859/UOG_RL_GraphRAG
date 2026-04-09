# =============================================================================
# src/utils/ollama_client.py
# Reliable HTTP helper for local Ollama API calls.
# =============================================================================
"""
Ollama HTTP Client
==================
Replaces bare ``requests.post`` calls throughout the codebase with a single
function that addresses the ``ReadTimeout`` / unreachable-server failures that
appear after 3-4 hours of continuous use.

Root-cause fixes applied
------------------------
1. **Tuple timeout** ``(connect_timeout, read_timeout)`` — a 10-second TCP
   connect deadline means a dead/hung Ollama process is detected immediately
   instead of waiting the full read-timeout before failing.
2. **No keep-alive / fresh session per attempt** — HTTP keep-alive lets
   ``requests`` reuse a pooled TCP connection.  After hours of idle time those
   connections silently go stale; the next write succeeds but the read hangs
   forever.  Calling ``session.headers["Connection"] = "close"`` and creating
   a fresh ``Session`` for every attempt prevents this entirely.
3. **Retry with exponential back-off** — transient ``ReadTimeout`` or
   ``ConnectionError`` (e.g. Ollama briefly unavailable after a GPU OOM) are
   retired up to ``max_retries`` times so the pipeline can recover without
   human intervention.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# TCP connection establishment deadline (seconds).  Short so a dead Ollama
# process is detected in seconds rather than minutes.
_CONNECT_TIMEOUT: int = 10


def ollama_post(
    url: str,
    payload: dict[str, Any],
    *,
    read_timeout: int = 300,
    max_retries: int = 3,
    initial_backoff: float = 5.0,
) -> dict[str, Any]:
    """POST *payload* to an Ollama endpoint and return the parsed JSON body.

    Parameters
    ----------
    url:
        Full endpoint URL, e.g. ``http://localhost:11434/api/generate``.
    payload:
        JSON-serialisable request body dict.
    read_timeout:
        Seconds to wait for the model to stream back a complete response.
        Passed through from ``settings.ollama_request_timeout`` so it remains
        user-configurable.
    max_retries:
        Number of *additional* attempts after an initial failure.  Set to 0
        to disable retry.
    initial_backoff:
        Seconds to wait before the first retry.  Doubles on each subsequent
        attempt, capped at 60 s.

    Returns
    -------
    dict
        Parsed JSON response from Ollama.

    Raises
    ------
    requests.exceptions.ReadTimeout
        Re-raised once ``max_retries`` is exhausted.
    requests.exceptions.ConnectionError
        Re-raised once ``max_retries`` is exhausted.
    requests.exceptions.HTTPError
        Raised immediately (without retry) for non-2xx HTTP responses, as
        those indicate a logic error rather than a transient failure.
    """
    import requests
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ReadTimeout

    retryable = (ReadTimeout, RequestsConnectionError)
    delay = initial_backoff

    for attempt in range(max_retries + 1):
        # A brand-new Session per attempt guarantees a fresh TCP connection.
        # "Connection: close" tells the server (and urllib3) not to pool the
        # socket — this is the primary defence against stale-connection hangs.
        with requests.Session() as session:
            session.headers.update({"Connection": "close"})
            try:
                resp = session.post(
                    url,
                    json=payload,
                    timeout=(_CONNECT_TIMEOUT, read_timeout),
                )
                resp.raise_for_status()
                return resp.json()
            except retryable as exc:
                if attempt == max_retries:
                    logger.error(
                        "Ollama request to %s failed after %d attempt(s). Last error: %s",
                        url,
                        attempt + 1,
                        exc,
                    )
                    raise

                logger.warning(
                    "Ollama connection error on attempt %d/%d: %s — "
                    "retrying in %.0f s with a fresh connection.",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 60.0)

    # Unreachable — the loop always raises or returns inside.
    raise RuntimeError("ollama_post: unexpected exit from retry loop")  # pragma: no cover
