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
from typing import Sequence

import numpy as np

from src.config import EmbeddingProvider, get_settings

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
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or get_settings().embedding_model_name
        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), show_progress_bar=False)


class OpenAIEmbedder(BaseEmbedder):
    """Embed via the OpenAI Embeddings API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        settings = get_settings()
        self.model = model or settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        logger.info("Using OpenAI embedding model: %s", self.model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self.client.embeddings.create(input=list(texts), model=self.model)
        return np.array([d.embedding for d in response.data])


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
