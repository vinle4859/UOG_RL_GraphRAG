# =============================================================================
# src/rag/embedder.py
# Embedding model abstraction layer.
# =============================================================================
"""
Embedder
========
Wraps different embedding backends behind a simple ``embed(texts)`` interface
so the rest of the pipeline is backend-agnostic.

Supported backends (configured via ``EMBEDDING_PROVIDER`` env var):
- ``sentence_transformers`` — local, free, fast.  Good default.
- ``openai`` — cloud-based, higher quality on some benchmarks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.config import EmbeddingProvider, get_settings
from src.utils.retry import openai_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Thin interface that all embedders must implement."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Parameters
        ----------
        texts : Sequence[str]
            Input strings.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(texts), embedding_dim)``.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed with a local Sentence-Transformers model."""

    def __init__(self, model_name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name == "httpx":
                raise RuntimeError(
                    "Missing dependency `httpx`, which is required by "
                    "`sentence-transformers` in this environment. "
                    "Install it with `pip install httpx` before running indexing or benchmarks."
                ) from exc
            raise

        self.model_name = model_name or get_settings().embedding_model_name
        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), show_progress_bar=False)


class OpenAIEmbedder(BaseEmbedder):
    """Embed via the OpenAI Embeddings API with retry and batching."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        settings = get_settings()
        self.model = model or settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.batch_size = settings.embedding_batch_size
        logger.info("Using OpenAI embedding model: %s", self.model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts in batches with automatic retry on rate-limit errors."""
        all_texts = list(texts)
        if not all_texts:
            return np.empty((0, 0))

        results: list[np.ndarray] = []
        for batch_start in range(0, len(all_texts), self.batch_size):
            batch = all_texts[batch_start : batch_start + self.batch_size]
            logger.debug(
                "Embedding batch %d–%d of %d texts",
                batch_start + 1,
                batch_start + len(batch),
                len(all_texts),
            )
            response = self._embed_batch_with_retry(batch)
            results.append(np.array([d.embedding for d in response.data]))

        return np.vstack(results)

    @openai_retry()
    def _embed_batch_with_retry(self, batch: list[str]):
        """Single batched embedding call, wrapped with retry logic."""
        return self.client.embeddings.create(input=batch, model=self.model)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder(provider: EmbeddingProvider | None = None) -> BaseEmbedder:
    """
    Create an embedder based on the configured provider.

    Parameters
    ----------
    provider : EmbeddingProvider, optional
        Override the setting from ``.env``.
    """
    provider = provider or get_settings().embedding_provider

    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder()
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
