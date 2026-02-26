# =============================================================================
# src/rag/vectorstore.py
# Vector store abstraction for indexing and similarity search.
# =============================================================================
"""
Vector Store
============
Provides a unified interface for storing and querying document chunk embeddings.

Supported backends:
- **ChromaDB** — simple, file-based, good for prototyping.
- **FAISS** — efficient, widely used, supports GPU acceleration.

Usage
-----
::

    store = get_vectorstore()
    store.add(chunk_ids, texts, embeddings)
    results = store.query(query_embedding, top_k=5)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import VectorStoreType, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single retrieval result with score."""
    chunk_id: str
    text: str
    score: float
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Interface every vector store backend must implement."""

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        """Insert documents into the store."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar documents."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed vectors."""
        ...


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

class ChromaVectorStore(BaseVectorStore):
    """Wrapper around ChromaDB for persistent vector storage."""

    def __init__(self, persist_dir: Path | None = None, collection_name: str = "rag_chunks"):
        import chromadb

        persist_dir = persist_dir or get_settings().vector_store_path
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' at %s (%d vectors)",
                     collection_name, persist_dir, self.collection.count())

    def add(self, ids, texts, embeddings, metadatas=None):
        self.collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=embeddings.tolist(),
            metadatas=list(metadatas) if metadatas else None,
        )

    def query(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        out: list[SearchResult] = []
        for cid, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0] if results["metadatas"] else [{}] * top_k,
        ):
            out.append(SearchResult(chunk_id=cid, text=doc, score=1 - dist, metadata=meta))
        return out

    def count(self):
        return self.collection.count()


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

class FAISSVectorStore(BaseVectorStore):
    """Wrapper around FAISS for in-memory / on-disk vector search."""

    def __init__(self):
        import faiss  # noqa: F401

        self._index = None  # Lazy-initialised on first add()
        self._id_map: dict[int, str] = {}
        self._text_map: dict[str, str] = {}
        self._meta_map: dict[str, dict] = {}
        self._next_int_id = 0

    def add(self, ids, texts, embeddings, metadatas=None):
        import faiss

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # Inner product (normalise for cosine)
            logger.info("Created FAISS index with dim=%d", dim)

        # Normalise for cosine similarity
        faiss.normalize_L2(embeddings)
        start = self._next_int_id
        self._index.add(embeddings)

        for i, (cid, txt) in enumerate(zip(ids, texts)):
            int_id = start + i
            self._id_map[int_id] = cid
            self._text_map[cid] = txt
            if metadatas:
                self._meta_map[cid] = metadatas[i]
        self._next_int_id = start + len(ids)

    def query(self, query_embedding, top_k=5):
        import faiss

        qe = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(qe)
        scores, indices = self._index.search(qe, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            cid = self._id_map[int(idx)]
            results.append(SearchResult(
                chunk_id=cid,
                text=self._text_map[cid],
                score=float(score),
                metadata=self._meta_map.get(cid),
            ))
        return results

    def count(self):
        return self._index.ntotal if self._index else 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vectorstore(store_type: VectorStoreType | None = None) -> BaseVectorStore:
    """Create a vector store based on configuration."""
    store_type = store_type or get_settings().vector_store_type

    if store_type == VectorStoreType.CHROMA:
        return ChromaVectorStore()
    elif store_type == VectorStoreType.FAISS:
        return FAISSVectorStore()
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
