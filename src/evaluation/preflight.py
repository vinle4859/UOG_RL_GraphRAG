# =============================================================================
# src/evaluation/preflight.py
# Fail-fast checks for benchmark and graph export workflows.
# =============================================================================

from __future__ import annotations

import importlib
import json
from pathlib import Path

import requests

from src.config import EmbeddingProvider, LLMProvider, VectorStoreType, get_settings
from src.graph_rag.community import (
    default_community_path,
    has_community_summaries,
    load_communities,
)


def ensure_dependency(module_name: str, install_hint: str) -> None:
    """Raise a helpful RuntimeError when a required import is unavailable."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency `{module_name}`. Install it with `{install_hint}` before running this command."
        ) from exc


def ensure_parquet_engine_available() -> None:
    """Validate that a parquet engine is installed for artifact export."""
    try:
        importlib.import_module("pyarrow")
    except ModuleNotFoundError:
        try:
            importlib.import_module("fastparquet")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "No parquet engine is installed. Install one with `pip install pyarrow` "
                "before running `rag-bench graphml-to-artifacts`."
            ) from exc


def run_benchmark_preflight(
    questions_path: Path | str,
    *,
    graphrag_modes: list[str] | None = None,
    llm_model: str | None = None,
    graph_path: Path | str = "data/graphs/knowledge_graph.graphml",
) -> list[str]:
    """Run fail-fast checks for benchmark safety and return a readable checklist."""
    settings = get_settings()
    notes: list[str] = []
    errors: list[str] = []
    question_path = Path(questions_path)
    if not question_path.exists():
        errors.append(f"Question file not found: {question_path}")
        raise RuntimeError("Benchmark preflight failed:\n- " + "\n- ".join(errors))

    try:
        questions = json.loads(question_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Question file is not valid JSON: {question_path} ({exc})")
        raise RuntimeError("Benchmark preflight failed:\n- " + "\n- ".join(errors)) from exc

    if not isinstance(questions, list) or not questions:
        errors.append(f"Question file is empty or not a JSON array: {question_path}")
        raise RuntimeError("Benchmark preflight failed:\n- " + "\n- ".join(errors))
    notes.append(f"Questions: {len(questions)} loaded from {question_path}")

    if settings.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        try:
            ensure_dependency("httpx", "pip install httpx")
            ensure_dependency(
                "sentence_transformers",
                "pip install sentence-transformers httpx",
            )
            notes.append("Embedding stack: sentence-transformers import check passed")
        except RuntimeError as exc:
            errors.append(str(exc))
    elif settings.embedding_provider == EmbeddingProvider.OPENAI:
        try:
            ensure_dependency("openai", "pip install openai")
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured for the selected embedding provider."
                )
            notes.append("Embedding stack: OpenAI configuration present")
        except RuntimeError as exc:
            errors.append(str(exc))

    if settings.vector_store_type == VectorStoreType.CHROMA:
        try:
            ensure_dependency("chromadb", "pip install chromadb")
            from src.rag.vectorstore import get_vectorstore

            vector_count = get_vectorstore().count()
            if vector_count <= 0:
                raise RuntimeError(
                    f"Vector store is empty at {settings.vector_store_path}. Run `rag-bench index` first."
                )
            notes.append(f"Vector store: Chroma reachable with {vector_count} indexed vectors")
        except RuntimeError as exc:
            errors.append(str(exc))
    else:
        try:
            ensure_dependency("faiss", "pip install faiss-cpu")
            notes.append("Vector store: FAISS dependency import check passed")
        except RuntimeError as exc:
            errors.append(str(exc))

    effective_model = llm_model or settings.llm_model_name or settings.ollama_model
    if settings.llm_provider == LLMProvider.OLLAMA:
        try:
            response = requests.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=(5, 30),
            )
            response.raise_for_status()
            payload = response.json()
            available = sorted(model.get("name", "") for model in payload.get("models", []))
            if effective_model not in available:
                preview = ", ".join(name for name in available[:8] if name)
                raise RuntimeError(
                    f"Ollama model `{effective_model}` is not available at {settings.ollama_base_url}. "
                    f"Available models include: {preview or '(none)'}"
                )
            notes.append(f"Ollama: model `{effective_model}` is available")
        except Exception as exc:
            errors.append(str(exc))
    elif settings.llm_provider == LLMProvider.OPENAI:
        try:
            ensure_dependency("openai", "pip install openai")
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured for the selected LLM provider."
                )
            notes.append("LLM provider: OpenAI configuration present")
        except RuntimeError as exc:
            errors.append(str(exc))

    graph_path = Path(graph_path)
    if not graph_path.exists():
        errors.append(
            f"Graph artifact not found: {graph_path}. Run `rag-bench build-graph` or `build-topology` first."
        )
    else:
        notes.append(f"Graph artifact: {graph_path} exists")

    modes = graphrag_modes or ["local", "global", "graph_only", "hybrid"]
    if any(mode in {"global", "hybrid"} for mode in modes):
        community_path = default_community_path(graph_path)
        if not community_path.exists():
            errors.append(
                f"Community sidecar not found: {community_path}. "
                "Run `rag-bench refresh-communities --summarise` first."
            )
        else:
            communities = load_communities(community_path)
            if not has_community_summaries(communities):
                errors.append(
                    f"Community sidecar exists but has no summaries: {community_path}. "
                    "Run `rag-bench refresh-communities --summarise` first."
                )
            else:
                notes.append(
                    f"Community sidecar: {community_path} loaded with {len(communities)} communities and summaries"
                )
    else:
        notes.append(f"Community summaries not required for modes: {', '.join(modes)}")

    try:
        importlib.import_module("dateutil")
        notes.append("Optional dependency: python-dateutil present")
    except ModuleNotFoundError:
        notes.append(
            "Optional dependency: python-dateutil missing (not required for benchmark runtime)"
        )

    if errors:
        raise RuntimeError("Benchmark preflight failed:\n- " + "\n- ".join(errors))

    return notes
