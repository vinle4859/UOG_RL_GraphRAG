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

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # tqdm is optional
    _tqdm = None  # type: ignore[assignment]

from src.config import get_settings
from src.data.chunker import Chunk, ChunkingPipeline
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
        self.generator = generator
        self.retriever = Retriever(embedder=self.embedder, vectorstore=self.vectorstore)
        self.chunker = ChunkingPipeline()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_path() -> Path:
        settings = get_settings()
        # Store in results_dir, NOT processed_data_dir, so the document
        # loader never mistakes the checkpoint JSON for a document.
        path = settings.results_dir / "index_checkpoint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _load_checkpoint() -> set[str]:
        """Return the set of doc_ids already successfully indexed."""
        cp = RAGPipeline._checkpoint_path()
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                return set(data.get("indexed_doc_ids", []))
            except Exception:
                logger.warning("Could not read checkpoint file — starting fresh.")
        return set()

    @staticmethod
    def _save_checkpoint(indexed_doc_ids: set[str]) -> None:
        """Persist the set of indexed doc_ids to disk."""
        cp = RAGPipeline._checkpoint_path()
        cp.write_text(
            json.dumps({"indexed_doc_ids": sorted(indexed_doc_ids)}, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        batch_size: int = 64,
        data_dir: str | None = None,
        resume: bool = True,
    ) -> int:
        """
        Load all documents, chunk them, embed, and store in the vector DB.

        Supports **resuming** interrupted runs: by default, already-indexed
        documents (tracked in ``data/processed/index_checkpoint.json``) are
        skipped so the pipeline continues from where it left off.

        Parameters
        ----------
        batch_size : int
            Number of chunks to embed in one go.
        data_dir : str, optional
            Override the default data directory.
        resume : bool
            If ``True`` (default), skip documents that were successfully
            indexed in a previous run.  Set to ``False`` to force a full
            re-index from scratch.

        Returns the total number of vectors now in the store.
        """
        docs = load_documents(data_dir)
        if not docs:
            logger.warning("No documents found — nothing to index.")
            return 0

        # ---- Resume logic ------------------------------------------------
        indexed_doc_ids: set[str] = self._load_checkpoint() if resume else set()

        if indexed_doc_ids:
            skipped = [d for d in docs if d.doc_id in indexed_doc_ids]
            docs = [d for d in docs if d.doc_id not in indexed_doc_ids]
            logger.info(
                "Resume mode: skipping %d already-indexed docs, %d remaining.",
                len(skipped),
                len(docs),
            )

        if not docs:
            total = self.vectorstore.count()
            logger.info("All documents already indexed — %d vectors in store.", total)
            return total

        # ---- Chunk -------------------------------------------------------
        chunks = self.chunker.run(docs)
        logger.info("Indexing %d chunks in batches of %d …", len(chunks), batch_size)

        # ---- Embed & store — cross-document flat batching ----------------
        # Batching across docs rather than within each doc maximises throughput:
        # fewer (larger) API calls, better GPU utilisation for local models.
        # We track a per-doc chunk count so we can still checkpoint at doc level.
        chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            chunks_by_doc[c.doc_id].append(c)

        # Preserve doc order for deterministic checkpointing
        ordered_doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))

        # Flatten into one stream, keeping track of boundaries
        flat_chunks = [c for doc_id in ordered_doc_ids for c in chunks_by_doc[doc_id]]

        total_batches = (len(flat_chunks) + batch_size - 1) // batch_size
        progress = (
            _tqdm(total=len(flat_chunks), unit="chunk", desc="Embedding & indexing")
            if _tqdm is not None
            else None
        )

        # Track how many chunks of each doc have been stored
        stored_counts: dict[str, int] = defaultdict(int)
        doc_total = {doc_id: len(chunks_by_doc[doc_id]) for doc_id in ordered_doc_ids}

        try:
            for i in range(0, len(flat_chunks), batch_size):
                batch: list[Chunk] = flat_chunks[i : i + batch_size]
                ids = [c.chunk_id for c in batch]
                texts = [c.text for c in batch]
                metadatas = [{"doc_id": c.doc_id, "index": c.index, **c.metadata} for c in batch]
                try:
                    embeddings = self.embedder.embed(texts)
                    self.vectorstore.add(ids, texts, embeddings, metadatas)
                except Exception:
                    logger.exception(
                        "Batch %d/%d failed — affected doc(s) will be retried next run.",
                        i // batch_size + 1,
                        total_batches,
                    )
                    if progress:
                        progress.update(len(batch))
                    continue

                if progress:
                    progress.update(len(batch))

                # Check if any doc is now fully stored → checkpoint it
                for c in batch:
                    stored_counts[c.doc_id] += 1
                    if stored_counts[c.doc_id] == doc_total[c.doc_id]:
                        indexed_doc_ids.add(c.doc_id)
                        self._save_checkpoint(indexed_doc_ids)
                        logger.debug("Checkpointed doc_id=%s", c.doc_id)
        finally:
            if progress:
                progress.close()

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
        self.generator = self.generator or get_generator()
        answer = self.generator.generate(question, retrieved)
        return RAGResult(answer=answer, retrieved_chunks=retrieved, query=question)
