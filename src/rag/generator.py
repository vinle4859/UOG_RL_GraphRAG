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

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def generate(self, query: str, context_chunks: list[SearchResult]) -> str:
        context_str = "\n\n".join(
            f"[{c.chunk_id}] {c.text}" for c in context_chunks
        )
        user_msg = RAG_USER_TEMPLATE.format(context=context_str, question=query)

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

    def __init__(self, model: str = "llama3"):
        import requests  # noqa: F401 — validate availability

        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = model

    def generate(self, query: str, context_chunks: list[SearchResult]) -> str:
        import requests

        context_str = "\n\n".join(
            f"[{c.chunk_id}] {c.text}" for c in context_chunks
        )
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"{RAG_USER_TEMPLATE.format(context=context_str, question=query)}"
        )

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_generator(provider: LLMProvider | None = None) -> BaseGenerator:
    """Create a generator based on configuration."""
    provider = provider or get_settings().llm_provider

    if provider == LLMProvider.OPENAI:
        return OpenAIGenerator()
    elif provider == LLMProvider.OLLAMA:
        return OllamaGenerator()
    else:
        raise NotImplementedError(f"Generator not yet implemented for: {provider}")
