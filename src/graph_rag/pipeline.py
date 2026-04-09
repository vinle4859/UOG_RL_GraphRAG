# =============================================================================
# src/graph_rag/pipeline.py
# End-to-end Graph RAG pipeline.
# =============================================================================
"""
Graph RAG Pipeline Orchestrator
===============================
Orchestrates the full GraphRAG flow:

1. Load and chunk documents.
2. Extract entities and relations from each chunk.
3. Build the knowledge graph.
4. Detect communities and optionally generate summaries.
5. Index chunks in the vector store (reuse from standard RAG).
6. At query time, use graph-augmented retrieval plus LLM generation.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.data.chunker import Chunk, ChunkingPipeline
from src.data.loader import load_documents
from src.graph_rag.community import (
    Community,
    default_community_path,
    detect_communities_leiden,
    has_community_summaries,
    load_communities,
    save_communities,
    summarise_community,
)
from src.graph_rag.entity_extractor import EntityExtractor, ExtractionResult
from src.graph_rag.graph_builder import KnowledgeGraph
from src.graph_rag.retriever import GraphRetriever, GraphSearchResult, SearchMode
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.generator import BaseGenerator, get_generator
from src.rag.vectorstore import BaseVectorStore, SearchResult, get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGResult:
    """Wraps a generated answer with GraphRAG-specific evidence."""

    answer: str
    query: str
    search_result: GraphSearchResult | None = None


class GraphRAGPipeline:
    """
    Full Graph RAG pipeline.

    Parameters
    ----------
    embedder : BaseEmbedder, optional
    vectorstore : BaseVectorStore, optional
    generator : BaseGenerator, optional
    graph_path : Path, optional
        Where to save and load the knowledge graph.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        generator: BaseGenerator | None = None,
        graph_path: Path | str = "data/graphs/knowledge_graph.graphml",
    ):
        self._embedder = embedder
        self._vectorstore = vectorstore
        self.generator = generator
        self.graph_path = Path(graph_path)
        self.community_path = default_community_path(self.graph_path)
        self._manifest_path = self.graph_path.parent / "indexed_docs.json"

        self.chunker = ChunkingPipeline()
        self.extractor = EntityExtractor()
        self.kg = KnowledgeGraph()
        self.communities: list[Community] = []
        self.retriever: GraphRetriever | None = None

    @property
    def embedder(self) -> BaseEmbedder:
        """Lazy-load the shared embedder only when retrieval/indexing needs it."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def vectorstore(self) -> BaseVectorStore:
        """Lazy-load the shared vector store only when retrieval/indexing needs it."""
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore()
        return self._vectorstore

    # ------------------------------------------------------------------
    # Build phase (offline)
    # ------------------------------------------------------------------

    def build(
        self,
        batch_size: int = 64,
        summarise: bool = True,
        summary_limit: int | None = None,
        limit: int | None = None,
        resume: bool = False,
        checkpoint_path: Path | str | None = None,
        extraction_workers: int | None = None,
    ) -> None:
        """
        Run the full build pipeline: chunk -> extract -> graph -> communities -> index.

        Parameters
        ----------
        batch_size : int
            Embedding batch size.
        summarise : bool
            Whether to generate community summaries.
        summary_limit : int, optional
            Cap the number of communities summarised. Communities are ordered
            by size descending before the limit is applied.
        limit : int, optional
            Cap the number of documents processed.
        resume : bool
            If True, skip documents already recorded in the indexed-docs manifest
            and load extraction results from the checkpoint file.
        checkpoint_path : Path, optional
            JSONL file for per-chunk extraction checkpointing. Defaults to
            ``data/graphs/extraction_checkpoint.jsonl`` when ``resume`` is True.
        extraction_workers : int, optional
            Number of concurrent extraction worker threads for LLM calls.
        """
        docs = load_documents()
        if not docs:
            logger.warning("No documents found - aborting build.")
            return

        seen: set[str] = set()
        docs = [doc for doc in docs if not (doc.doc_id in seen or seen.add(doc.doc_id))]  # type: ignore[func-returns-value]
        logger.info("Loaded %d unique documents.", len(docs))

        if resume:
            already_indexed = self._load_manifest()
            if already_indexed:
                before = len(docs)
                docs = [doc for doc in docs if doc.doc_id not in already_indexed]
                logger.info(
                    "Resume: skipping %d already-indexed docs, %d remaining.",
                    before - len(docs),
                    len(docs),
                )

        if limit is not None:
            docs = docs[:limit]
            logger.info("Limit applied: processing %d documents.", len(docs))

        if not docs:
            logger.info("Nothing new to process.")
            return

        chunks = self.chunker.run(docs)

        checkpoint = None
        if resume:
            checkpoint = checkpoint_path or self.graph_path.parent / "extraction_checkpoint.jsonl"
        elif checkpoint_path:
            checkpoint = Path(checkpoint_path)

        logger.info("Extracting entities from %d chunks...", len(chunks))
        extraction_inputs = [(chunk.chunk_id, chunk.text) for chunk in chunks]
        results = self.extractor.extract_batch(
            extraction_inputs,
            checkpoint_path=checkpoint,
            num_workers=extraction_workers,
        )

        self.kg.add_extractions(results)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg.save(self.graph_path)
        self._refresh_communities_from_graph(
            summarise=summarise,
            summary_limit=summary_limit,
        )

        self._index_chunks(chunks, batch_size)

        indexed = self._load_manifest()
        indexed.update(doc.doc_id for doc in docs)
        self._save_manifest(indexed)

        self.retriever = GraphRetriever(
            graph=self.kg.graph,
            communities=self.communities,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        logger.info("GraphRAG build complete.")

    # ------------------------------------------------------------------
    # Topology-only build from existing checkpoint (safe for concurrent use)
    # ------------------------------------------------------------------

    def build_topology_from_checkpoint(
        self,
        checkpoint_path: Path | str,
        limit: int | None = None,
        summarise: bool = True,
        summary_limit: int | None = None,
    ) -> None:
        """Build graph topology from a pre-existing extraction checkpoint JSONL."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        snapshot_path = checkpoint_path.with_suffix(".snapshot.jsonl")
        logger.info("Snapshotting checkpoint -> %s", snapshot_path)
        shutil.copy2(checkpoint_path, snapshot_path)

        good_statuses = {"ok", "fallback_success"}
        loaded = skipped_status = parse_errors = 0
        total_entities = total_relations = 0
        results: list[ExtractionResult] = []

        try:
            with snapshot_path.open(encoding="utf-8") as handle:
                for lineno, raw_line in enumerate(handle, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        parse_errors += 1
                        logger.warning(
                            "Line %d: JSON parse error while reading checkpoint snapshot - %s",
                            lineno,
                            exc,
                        )
                        continue

                    status = payload.get("status", "ok")
                    if status not in good_statuses:
                        skipped_status += 1
                        continue

                    try:
                        result = EntityExtractor._deserialize(payload)
                    except Exception as exc:
                        parse_errors += 1
                        logger.warning(
                            "Line %d (chunk_id=%r): deserialization failed - %s",
                            lineno,
                            payload.get("chunk_id", "?"),
                            exc,
                        )
                        continue

                    total_entities += len(result.entities)
                    total_relations += len(result.relations)
                    results.append(result)
                    loaded += 1

                    if limit is not None and loaded >= limit:
                        logger.info("Limit of %d chunks reached - stopping read.", limit)
                        break
        finally:
            snapshot_path.unlink(missing_ok=True)
            logger.debug("Snapshot removed: %s", snapshot_path)

        logger.info(
            "Checkpoint read: %d loaded, %d skipped (non-ok status), %d parse errors.",
            loaded,
            skipped_status,
            parse_errors,
        )
        logger.info(
            "Entities loaded: %d   Relations loaded: %d",
            total_entities,
            total_relations,
        )

        if not results:
            logger.warning("No valid extraction results found - graph will be empty.")

        logger.info("Building graph from %d extraction results...", len(results))
        self.kg.add_extractions(results)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg.save(self.graph_path)
        logger.info(
            "Graph saved -> %s (%d nodes, %d edges).",
            self.graph_path,
            self.kg.graph.number_of_nodes(),
            self.kg.graph.number_of_edges(),
        )

        self._refresh_communities_from_graph(
            summarise=summarise,
            summary_limit=summary_limit,
        )
        self.retriever = GraphRetriever(
            graph=self.kg.graph,
            communities=self.communities,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        logger.info("build-topology complete - graph ready for querying.")

    def _index_chunks(self, chunks: list[Chunk], batch_size: int) -> None:
        """Embed and index chunks in the vector store."""
        logger.info("Indexing %d chunks...", len(chunks))
        batches = range(0, len(chunks), batch_size)
        for start in tqdm(batches, desc="Indexing chunks", unit="batch"):
            batch = chunks[start : start + batch_size]
            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [{"doc_id": chunk.doc_id, "index": chunk.index} for chunk in batch]
            embeddings = self.embedder.embed(texts)
            self.vectorstore.add(ids, texts, embeddings, metadatas)

    # ------------------------------------------------------------------
    # Manifest helpers (incremental indexing)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> set[str]:
        """Return the set of doc_ids already fully indexed."""
        if self._manifest_path.exists():
            try:
                return set(json.loads(self._manifest_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return set()

    def _save_manifest(self, doc_ids: set[str]) -> None:
        """Persist the set of indexed doc_ids to disk."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(sorted(doc_ids), indent=2), encoding="utf-8")

    def _refresh_communities_from_graph(
        self,
        *,
        summarise: bool,
        summary_limit: int | None = None,
        summary_model: str | None = None,
    ) -> None:
        """Detect, optionally summarise, and persist communities for the loaded graph."""
        self.communities = detect_communities_leiden(self.kg.graph)
        logger.info("Detected %d communities.", len(self.communities))

        if summarise:
            ordered = sorted(
                self.communities, key=lambda community: len(community.nodes), reverse=True
            )
            selected = ordered[:summary_limit] if summary_limit is not None else ordered
            logger.info(
                "Summarising %d communities%s...",
                len(selected),
                f" (limit={summary_limit})" if summary_limit is not None else "",
            )
            for community in tqdm(selected, desc="Summarising communities", unit="community"):
                try:
                    summarise_community(
                        community,
                        self.kg.graph,
                        model=summary_model or "gpt-4o-mini",
                    )
                except Exception:
                    logger.exception("Failed to summarise community %d", community.community_id)
        else:
            logger.info("Community summarisation skipped.")

        save_communities(self.communities, self.community_path, graph_path=self.graph_path)

    # ------------------------------------------------------------------
    # Query phase (online)
    # ------------------------------------------------------------------

    def refresh_communities(
        self,
        *,
        summarise: bool = True,
        summary_limit: int | None = None,
        summary_model: str | None = None,
    ) -> None:
        """Refresh the community sidecar from an existing graph."""
        self.kg.load(self.graph_path)
        self._refresh_communities_from_graph(
            summarise=summarise,
            summary_limit=summary_limit,
            summary_model=summary_model,
        )
        self.retriever = None

    def load_graph(self, *, require_summaries: bool = False) -> None:
        """Load a previously built graph and initialise the retriever."""
        self.kg.load(self.graph_path)
        if self.community_path.exists():
            self.communities = load_communities(self.community_path)
        else:
            logger.warning(
                "Community sidecar missing at %s - re-detecting communities from GraphML.",
                self.community_path,
            )
            self.communities = detect_communities_leiden(self.kg.graph)

        if require_summaries and not has_community_summaries(self.communities):
            raise RuntimeError(
                "Community summaries are required for global or hybrid GraphRAG modes. "
                "Run `rag-bench refresh-communities --summarise` first."
            )

        self.retriever = GraphRetriever(
            graph=self.kg.graph,
            communities=self.communities,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

    def query(
        self,
        question: str,
        mode: SearchMode = SearchMode.LOCAL,
        top_k: int | None = None,
    ) -> GraphRAGResult:
        """
        Retrieve context (graph-augmented) and generate an answer.

        Parameters
        ----------
        question : str
            User's question.
        mode : SearchMode
            Retrieval mode.
        top_k : int, optional
            Number of chunks for local search.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Call build(), refresh_communities(), or load_graph() before querying."
            )

        search_result = self.retriever.retrieve(question, mode=mode, top_k=top_k)
        generation_chunks: list[SearchResult] = list(search_result.chunk_results)

        if search_result.graph_context.strip():
            generation_chunks.append(
                SearchResult(
                    chunk_id="graph_context",
                    text=search_result.graph_context,
                    score=0.0,
                    metadata={"source": "graph_relations"},
                )
            )

        for idx, summary in enumerate(search_result.community_summaries, start=1):
            if summary.strip():
                generation_chunks.append(
                    SearchResult(
                        chunk_id=f"community_summary_{idx}",
                        text=summary,
                        score=0.0,
                        metadata={"source": "community_summary", "index": idx},
                    )
                )

        self.generator = self.generator or get_generator()
        answer = self.generator.generate(question, generation_chunks)
        return GraphRAGResult(answer=answer, query=question, search_result=search_result)
