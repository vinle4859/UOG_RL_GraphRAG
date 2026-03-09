# =============================================================================
# src/graph_rag/pipeline.py
# End-to-end Graph RAG pipeline.
# =============================================================================
"""
Graph RAG Pipeline Orchestrator
================================
Orchestrates the full GraphRAG flow:

1. Load & chunk documents.
2. Extract entities & relations from each chunk.
3. Build the knowledge graph.
4. Detect communities and generate summaries.
5. Index chunks in the vector store (reuse from standard RAG).
6. At query time, use graph-augmented retrieval + LLM generation.

Example
-------
::

    from src.graph_rag.pipeline import GraphRAGPipeline

    pipe = GraphRAGPipeline()
    pipe.build()                            # one-time: full build
    answer = pipe.query("What is PPO?")     # query time
    print(answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.data.chunker import Chunk, ChunkingPipeline, TokenChunker
from src.data.loader import load_documents
from src.graph_rag.community import Community, detect_communities_leiden, summarise_community
from src.graph_rag.entity_extractor import EntityExtractor
from src.graph_rag.graph_builder import KnowledgeGraph
from src.graph_rag.retriever import GraphRetriever, GraphSearchResult, SearchMode
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.generator import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, BaseGenerator, get_generator
from src.rag.vectorstore import BaseVectorStore, get_vectorstore

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
        Where to save/load the knowledge graph.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        generator: BaseGenerator | None = None,
        graph_path: Path | str = "data/graphs/knowledge_graph.graphml",
    ):
        self.embedder = embedder or get_embedder()
        self.vectorstore = vectorstore or get_vectorstore()
        self.generator = generator or get_generator()
        self.graph_path = Path(graph_path)

        self.chunker = ChunkingPipeline()
        self.extractor = EntityExtractor()
        self.kg = KnowledgeGraph()
        self.communities: list[Community] = []
        self.retriever: GraphRetriever | None = None

    # ------------------------------------------------------------------
    # Build phase (offline)
    # ------------------------------------------------------------------

    def build(self, batch_size: int = 64, summarise: bool = True) -> None:
        """
        Run the full build pipeline: chunk → extract → graph → communities
        → index.

        Parameters
        ----------
        batch_size : int
            Embedding batch size.
        summarise : bool
            Whether to generate community summaries (requires LLM calls).
        """
        # Step 1: Load & chunk
        docs = load_documents()
        if not docs:
            logger.warning("No documents found — aborting build.")
            return
        chunks = self.chunker.run(docs)

        # Step 2: Entity extraction
        logger.info("Extracting entities from %d chunks …", len(chunks))
        extraction_inputs = [(c.chunk_id, c.text) for c in chunks]
        results = self.extractor.extract_batch(extraction_inputs)

        # Step 3: Build graph
        self.kg.add_extractions(results)
        self.kg.save(self.graph_path)

        # Step 4: Community detection
        self.communities = detect_communities_leiden(self.kg.graph)
        if summarise:
            logger.info("Summarising %d communities …", len(self.communities))
            for comm in self.communities:
                try:
                    summarise_community(comm, self.kg.graph)
                except Exception:
                    logger.exception("Failed to summarise community %d", comm.community_id)

        # Step 5: Index chunks in vector store
        self._index_chunks(chunks, batch_size)

        # Step 6: Initialise retriever
        self.retriever = GraphRetriever(
            graph=self.kg.graph,
            communities=self.communities,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

        logger.info("GraphRAG build complete.")

    def _index_chunks(self, chunks: list[Chunk], batch_size: int) -> None:
        """Embed and index chunks in the vector store."""
        logger.info("Indexing %d chunks …", len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [{"doc_id": c.doc_id, "index": c.index} for c in batch]
            embeddings = self.embedder.embed(texts)
            self.vectorstore.add(ids, texts, embeddings, metadatas)

    # ------------------------------------------------------------------
    # Query phase (online)
    # ------------------------------------------------------------------

    def load_graph(self) -> None:
        """Load a previously built graph and initialise the retriever."""
        self.kg.load(self.graph_path)
        # Re-detect communities from the loaded graph
        self.communities = detect_communities_leiden(self.kg.graph)
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
            "local", "global", or "hybrid".
        top_k : int, optional
            Number of chunks for local search.

        Returns
        -------
        GraphRAGResult
        """
        if self.retriever is None:
            raise RuntimeError("Call build() or load_graph() before querying.")

        search_result = self.retriever.retrieve(question, mode=mode, top_k=top_k)

        # Assemble context for the LLM
        context_parts: list[str] = []

        # From vector search chunks
        for c in search_result.chunk_results:
            context_parts.append(f"[{c.chunk_id}] {c.text}")

        # From graph structure
        if search_result.graph_context:
            context_parts.append(f"\n--- Graph Relations ---\n{search_result.graph_context}")

        # From community summaries
        for i, summary in enumerate(search_result.community_summaries):
            context_parts.append(f"\n--- Community {i + 1} Summary ---\n{summary}")

        context_str = "\n\n".join(context_parts)
        user_msg = RAG_USER_TEMPLATE.format(context=context_str, question=question)

        # Generate answer via the configured generator (OpenAI / Ollama / etc.)
        answer = self.generator.generate(user_msg, search_result.chunk_results)

        return GraphRAGResult(answer=answer, query=question, search_result=search_result)
