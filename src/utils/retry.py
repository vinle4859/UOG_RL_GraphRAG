# =============================================================================
# src/utils/retry.py
# Retry helpers for external API calls (OpenAI, etc.)
# =============================================================================
"""
Retry Utilities
===============
Provides a decorator and a context-manager-friendly helper that retry
callables on transient errors (rate-limit / quota exceeded / connection
issues) with **exponential backoff + full jitter**.

Usage
-----
::

    from src.utils.retry import openai_retry

    @openai_retry()
    def my_api_call():
        ...

Or apply ad-hoc::

    result = openai_retry()(my_fn)(arg1, arg2)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Sequence, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retryable exception types
# ---------------------------------------------------------------------------

def _get_retryable_openai_exceptions() -> tuple[Type[BaseException], ...]:
    """Return OpenAI exception types that warrant a retry."""
    try:
        import openai
        return (
            openai.RateLimitError,       # HTTP 429 — rate limit / quota
            openai.APIConnectionError,   # network hiccup
            openai.APITimeoutError,      # request timed out
            openai.InternalServerError,  # transient 5xx
        )
    except ImportError:
        return (OSError,)


# ---------------------------------------------------------------------------
# Core retry implementation
# ---------------------------------------------------------------------------

def with_retry(
    func: Callable,
    *,
    max_retries: int = 6,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Sequence[Type[BaseException]] | None = None,
    jitter: bool = True,
):
    """
    Call *func* and retry on transient errors using exponential back-off.

    Parameters
    ----------
    func :
        Zero-argument callable to invoke.
    max_retries :
        Maximum number of retry attempts after the first failure.
    initial_backoff :
        Seconds to wait before the first retry.
    max_backoff :
        Upper bound on backoff delay (seconds).
    backoff_factor :
        Multiplier applied to the delay after each failure.
    retryable_exceptions :
        Exception types that trigger a retry.  Defaults to common OpenAI
        transient errors.
    jitter :
        If ``True``, applies full jitter (random value in [0, delay]) to
        avoid thundering-herd problems.

    Returns
    -------
    Any
        The return value of ``func`` on success.

    Raises
    ------
    Exception
        Re-raises the last exception once ``max_retries`` is exhausted.
    """
    if retryable_exceptions is None:
        retryable_exceptions = _get_retryable_openai_exceptions()

    exc_tuple = tuple(retryable_exceptions)
    delay = initial_backoff

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exc_tuple as exc:
            if attempt == max_retries:
                logger.error(
                    "API call failed after %d attempts. Last error: %s",
                    max_retries + 1, exc,
                )
                raise

            actual_delay = random.uniform(0, delay) if jitter else delay
            actual_delay = min(actual_delay, max_backoff)

            logger.warning(
                "Transient API error (attempt %d/%d): %s. "
                "Retrying in %.1f s…",
                attempt + 1, max_retries + 1, exc, actual_delay,
            )
            time.sleep(actual_delay)
            delay = min(delay * backoff_factor, max_backoff)


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------

def openai_retry(
    max_retries: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    backoff_factor: float | None = None,
):
    """
    Decorator that wraps a function with :func:`with_retry`.

    Values default to the project ``Settings`` (loaded lazily so the
    decorator can be used at import time before settings are configured).

    Example
    -------
    ::

        @openai_retry()
        def call_api():
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve settings lazily to avoid circular imports at module load
            from src.config import get_settings
            cfg = get_settings()

            return with_retry(
                lambda: func(*args, **kwargs),
                max_retries=max_retries if max_retries is not None else cfg.api_max_retries,
                initial_backoff=initial_backoff if initial_backoff is not None else cfg.api_initial_backoff,
                max_backoff=max_backoff if max_backoff is not None else cfg.api_max_backoff,
                backoff_factor=backoff_factor if backoff_factor is not None else cfg.api_backoff_factor,
            )
        return wrapper
    return decorator
