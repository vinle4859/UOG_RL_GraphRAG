# =============================================================================
# src/rag/pipeline.py
# End-to-end standard RAG pipeline: ingest → index → query.
# =============================================================================
"""
RAG Pipeline Orchestrator
=========================
High-level API that ties together data loading, chunking, embedding, indexing,
retrieval, and generation into simple ``index()`` and ``query()`` calls.

Example
-------
::

    from src.rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    pipe.index()                            # one-time: process & embed all docs
    answer = pipe.query("What is PPO?")     # query time
    print(answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.chunker import Chunk, TokenChunker
from src.data.loader import load_documents
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.generator import BaseGenerator, get_generator
from src.rag.retriever import Retriever
from src.rag.vectorstore import BaseVectorStore, SearchResult, get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Wraps a generated answer together with the retrieval evidence."""
    answer: str
    retrieved_chunks: list[SearchResult]
    query: str


class RAGPipeline:
    """
    Full standard RAG pipeline.

    Parameters
    ----------
    embedder : BaseEmbedder, optional
    vectorstore : BaseVectorStore, optional
    generator : BaseGenerator, optional
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        generator: BaseGenerator | None = None,
    ):
        self.embedder = embedder or get_embedder()
        self.vectorstore = vectorstore or get_vectorstore()
        self.generator = generator or get_generator()
        self.retriever = Retriever(embedder=self.embedder, vectorstore=self.vectorstore)
        self.chunker = TokenChunker()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, batch_size: int = 64, data_dir: str | None = None) -> int:
        """
        Load all documents, chunk them, embed, and store in the vector DB.

        Parameters
        ----------
        batch_size : int
            Number of chunks to embed in one go.
        data_dir : str, optional
            Override the default data directory.

        Returns the total number of indexed chunks.
        """
        docs = load_documents(data_dir)
        if not docs:
            logger.warning("No documents found — nothing to index.")
            return 0

        chunks = self.chunker.chunk_documents(docs)
        logger.info("Indexing %d chunks in batches of %d …", len(chunks), batch_size)

        for i in range(0, len(chunks), batch_size):
            batch: list[Chunk] = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [{"doc_id": c.doc_id, "index": c.index, **c.metadata} for c in batch]
            embeddings = self.embedder.embed(texts)
            self.vectorstore.add(ids, texts, embeddings, metadatas)

        total = self.vectorstore.count()
        logger.info("Indexing complete — %d vectors in store.", total)
        return total

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int | None = None) -> RAGResult:
        """
        Retrieve relevant chunks and generate an answer.

        Parameters
        ----------
        question : str
            The user's natural-language question.
        top_k : int, optional
            Number of chunks to retrieve.

        Returns
        -------
        RAGResult
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, retrieved)
        return RAGResult(answer=answer, retrieved_chunks=retrieved, query=question)
