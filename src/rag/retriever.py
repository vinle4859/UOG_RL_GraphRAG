# =============================================================================
# src/rag/retriever.py
# Query-time retrieval from the vector store.
# =============================================================================
"""
Retriever
=========
Encapsulates the query flow: embed the user question → search the vector
store → return ranked chunks.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.config import get_settings
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.vectorstore import BaseVectorStore, SearchResult, get_vectorstore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieve relevant chunks for a natural-language query.

    Parameters
    ----------
    embedder : BaseEmbedder, optional
    vectorstore : BaseVectorStore, optional
    top_k : int, optional
        Number of chunks to return per query.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        top_k: int | None = None,
    ):
        self.embedder = embedder or get_embedder()
        self.vectorstore = vectorstore or get_vectorstore()
        self.top_k = top_k or get_settings().top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Embed *query* and return the top-k matching chunks.

        Parameters
        ----------
        query : str
            User question or search string.
        top_k : int, optional
            Override instance default.

        Returns
        -------
        list[SearchResult]
            Ranked retrieval results.
        """
        k = top_k or self.top_k
        query_emb = self.embedder.embed([query])[0]
        results = self.vectorstore.query(query_emb, top_k=k)
        logger.info("Retrieved %d chunks for query: %.80s…", len(results), query)
        return results

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[SearchResult]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]
