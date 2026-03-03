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
from typing import Optional

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
    openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment_name: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # --- Embedding ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model_name: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Vector Store ---
    vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    vector_store_path: Path = Path("./data/embeddings/chroma_db")

    # --- Graph Store ---
    graph_store_type: GraphStoreType = GraphStoreType.NETWORKX
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None

    # --- Paths ---
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    results_dir: Path = Path("./results")

    # --- Chunking ---
    chunk_size: int = Field(default=512, description="Token count per chunk")
    chunk_overlap: int = Field(default=64, description="Token overlap between chunks")

    # --- Retrieval ---
    top_k: int = Field(default=5, description="Number of retrieved chunks for RAG")

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton-like via lru_cache)."""
    return Settings()
