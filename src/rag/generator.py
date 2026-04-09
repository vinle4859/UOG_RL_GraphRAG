# =============================================================================
# src/rag/generator.py
# LLM answer generation given retrieved context.
# =============================================================================
"""
Generator
=========
Sends the retrieved context + user query to an LLM and returns the answer.
Supports OpenAI, Azure OpenAI, and Ollama backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.config import LLMProvider, get_settings
from src.rag.vectorstore import SearchResult
from src.utils.retry import openai_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use ONLY the provided context "
    "to answer the question. If the context does not contain enough "
    "information, say so. Cite the source chunk IDs when possible."
)

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------


class BaseGenerator(ABC):
    """Generate an answer from context + query."""

    @abstractmethod
    def generate(self, query: str, context_chunks: list[SearchResult]) -> str:
        """Return a natural-language answer."""
        ...


# ---------------------------------------------------------------------------
# OpenAI Generator
# ---------------------------------------------------------------------------


class OpenAIGenerator(BaseGenerator):
    """Generate answers using the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.llm_model_name

    def generate(self, query: str, context_chunks: list[SearchResult]) -> str:
        context_str = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context_chunks)
        user_msg = RAG_USER_TEMPLATE.format(context=context_str, question=query)
        return self._chat_with_retry(user_msg)

    @openai_retry()
    def _chat_with_retry(self, user_msg: str) -> str:
        """Single chat completion call, wrapped with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Ollama Generator (local models)
# ---------------------------------------------------------------------------


class OllamaGenerator(BaseGenerator):
    """Generate answers using a local Ollama server."""

    def __init__(self, model: str | None = None):
        import requests  # noqa: F401 — validate availability

        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = model or settings.llm_model_name
        logger.info("Using Ollama model: %s at %s", self.model, self.base_url)

    def generate(self, query: str, context_chunks: list[SearchResult]) -> str:
        from src.utils.ollama_client import ollama_post

        context_str = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context_chunks)
        settings = get_settings()
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"{RAG_USER_TEMPLATE.format(context=context_str, question=query)}"
        )

        data = ollama_post(
            f"{self.base_url}/api/generate",
            payload={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.ollama_generation_temperature,
                    "seed": settings.ollama_generation_seed,
                },
            },
            read_timeout=settings.ollama_request_timeout,
        )
        return data["response"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_generator(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> BaseGenerator:
    """Create a generator based on configuration."""
    provider = provider or get_settings().llm_provider

    if provider == LLMProvider.OPENAI:
        return OpenAIGenerator(model=model)
    elif provider == LLMProvider.OLLAMA:
        return OllamaGenerator(model=model)
    else:
        raise NotImplementedError(f"Generator not yet implemented for: {provider}")
