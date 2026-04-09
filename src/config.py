# =============================================================================
# src/config.py
# Centralised configuration loaded from environment variables / .env file.
# =============================================================================
"""
Configuration Management
========================
Uses pydantic-settings to validate and type-check all env vars at startup.
Copy ``.env.example`` → ``.env`` and fill in your keys before running.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enums for validated choices
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


class GraphStoreType(str, Enum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Project-wide settings.  Values are loaded from environment variables
    (prefixed or unprefixed) and the ``.env`` file at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model_name: str = "gpt-4o-mini"  # model name for the active provider
    openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment_name: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Fallback model used when the primary model fails after repair retry.
    # Set to a larger model (e.g. "llama3.2:3b") in .env as OLLAMA_FALLBACK_MODEL.
    # Leave unset (empty string) to disable fallback escalation.
    ollama_fallback_model: str | None = None

    # --- Embedding ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model_name: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Vector Store ---
    vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    vector_store_path: Path = Path("./data/embeddings/chroma_db")

    # --- Graph Store ---
    graph_store_type: GraphStoreType = GraphStoreType.NETWORKX
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    # --- Paths ---
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    results_dir: Path = Path("./results")

    # --- Chunking ---
    chunk_size: int = Field(default=512, description="Token count per chunk")
    chunk_overlap: int = Field(default=64, description="Token overlap between chunks")

    # --- Retrieval ---
    top_k: int = Field(default=5, description="Number of retrieved chunks for RAG")

    # --- Ollama ---
    ollama_request_timeout: int = Field(
        default=300, description="HTTP read timeout (seconds) for Ollama generate requests"
    )
    ollama_generation_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for Ollama answer generation during benchmark/query runs",
    )
    ollama_generation_seed: int = Field(
        default=42,
        description="Deterministic seed for Ollama answer generation during benchmark/query runs",
    )
    ollama_judge_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for Ollama scalar judge prompts",
    )
    ollama_judge_seed: int = Field(
        default=42,
        description="Deterministic seed for Ollama scalar judge prompts",
    )
    ollama_judge_num_predict: int = Field(
        default=16,
        ge=1,
        description="Maximum generated tokens for Ollama scalar judge prompts",
    )
    ollama_extraction_workers: int = Field(
        default=1,
        ge=1,
        description="Concurrent worker threads for GraphRAG entity extraction with Ollama",
    )
    benchmark_judge_workers: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent worker threads for LLM-as-judge scoring within one benchmark record",
    )

    # --- API Retry / Rate-limit handling ---
    api_max_retries: int = Field(
        default=6, description="Max retry attempts on quota/rate-limit errors"
    )
    api_initial_backoff: float = Field(default=1.0, description="Initial backoff delay in seconds")
    api_max_backoff: float = Field(default=60.0, description="Maximum backoff delay in seconds")
    api_backoff_factor: float = Field(default=2.0, description="Exponential backoff multiplier")
    embedding_batch_size: int = Field(
        default=100, description="Max texts per OpenAI embedding request"
    )

    # --- Performance ---
    num_workers: int = Field(
        default=4, description="Worker threads for parallel chunking/preprocessing"
    )

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton-like via lru_cache)."""
    return Settings()
