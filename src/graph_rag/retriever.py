# =============================================================================
# src/graph_rag/retriever.py
# Graph-augmented retrieval for GraphRAG.
# =============================================================================
"""
Graph-Augmented Retriever
=========================
Combines **vector similarity** with **graph-structural signals** to produce
a richer retrieval context than standard RAG.

Retrieval Strategies
--------------------
1. **Local Search** - Embed the query, retrieve top-k chunks from the vector
   store, then *expand* each chunk with its entity neighbours from the graph.
2. **Global Search** - Match the query against community summaries to find
   the most relevant community, then surface its member entities and their
   relations.
3. **Graph-Only Search** - Pure graph retrieval with NO vector store.
   Embeds entity labels, finds closest entities to the query, and expands
   by hops. This mode enables a fair benchmark against standard RAG by
   isolating the graph's contribution without overlapping vector search.
4. **Hybrid** - Combine both local and global search results.

These strategies mirror and extend the "local" and "global" search modes
described in Microsoft's GraphRAG paper.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import numpy as np

from src.graph_rag.community import Community
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.vectorstore import BaseVectorStore, SearchResult, get_vectorstore

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    GRAPH_ONLY = "graph_only"


@dataclass
class GraphSearchResult:
    """Extended search result with graph context."""

    chunk_results: list[SearchResult] = field(default_factory=list)
    graph_context: str = ""
    community_summaries: list[str] = field(default_factory=list)


class GraphRetriever:
    """
    Retriever that augments vector search with knowledge-graph traversal.

    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    communities : list[Community]
        Pre-detected community structures with summaries.
    embedder : BaseEmbedder, optional
    vectorstore : BaseVectorStore, optional
    top_k : int
        Number of chunks to retrieve in local search.
    expansion_hops : int
        How many hops to expand from each entity in local search.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        communities: list[Community] | None = None,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        top_k: int = 5,
        expansion_hops: int = 1,
    ):
        self.graph = graph
        self.communities = communities or []
        self.embedder = embedder or get_embedder()
        self.vectorstore = vectorstore or get_vectorstore()
        self.top_k = top_k
        self.expansion_hops = expansion_hops

        # Cache expensive lookup structures once per process.
        self._chunk_entity_index = self._build_chunk_entity_index()
        self._node_keys = list(self.graph.nodes())
        self._node_labels = [self.graph.nodes[node].get("label", node) for node in self._node_keys]
        self._node_label_embeddings: np.ndarray | None = None
        self._summary_records: list[tuple[int, str]] = []
        self._summary_embeddings: np.ndarray | None = None
        self._retrieval_cache: dict[tuple[str, str, int | None], GraphSearchResult] = {}

    # ------------------------------------------------------------------
    # Local Search
    # ------------------------------------------------------------------

    def local_search(self, query: str, top_k: int | None = None) -> GraphSearchResult:
        """
        Vector search + graph expansion.

        Steps:
        1. Embed query -> top-k chunks from vector store.
        2. For each chunk, look up entities mentioned (via chunk_id provenance).
        3. Expand entities by N hops to gather related context.
        """
        k = top_k or self.top_k
        query_emb = self.embedder.embed([query])[0]
        chunk_results = self.vectorstore.query(query_emb, top_k=k)

        related_entities: set[str] = set()
        for result in chunk_results:
            related_entities.update(self._chunk_entity_index.get(result.chunk_id, ()))

        expanded: set[str] = set()
        for entity in related_entities:
            ego = nx.ego_graph(self.graph, entity, radius=self.expansion_hops)
            expanded.update(ego.nodes())

        graph_context = self._build_graph_context(expanded)
        return GraphSearchResult(chunk_results=chunk_results, graph_context=graph_context)

    # ------------------------------------------------------------------
    # Global Search
    # ------------------------------------------------------------------

    def global_search(self, query: str, top_n: int = 3) -> GraphSearchResult:
        """
        Match the query against community summaries.

        Steps:
        1. Embed the query and each community summary.
        2. Rank communities by cosine similarity.
        3. Return the top-N community summaries as context.
        """
        if not self.communities:
            logger.warning("No communities available for global search.")
            return GraphSearchResult()

        self._ensure_summary_embeddings()
        if not self._summary_records or self._summary_embeddings is None:
            logger.warning("Community summaries are empty - run summarisation first.")
            return GraphSearchResult()

        query_emb = self.embedder.embed([query])[0]
        sims = np.dot(self._summary_embeddings, query_emb) / (
            np.linalg.norm(self._summary_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argsort(sims)[-min(top_n, len(sims)) :][::-1]
        top_summaries = [self._summary_records[i][1] for i in top_indices]
        return GraphSearchResult(community_summaries=top_summaries)

    # ------------------------------------------------------------------
    # Graph-Only Search
    # ------------------------------------------------------------------

    def graph_only_search(self, query: str, top_k: int | None = None) -> GraphSearchResult:
        """
        Pure graph retrieval - NO vector store involved.

        Steps:
        1. Embed the query and find the closest entity by comparing
           the query embedding with entity label embeddings.
        2. Expand from those seed entities by N hops.
        3. Return graph context (relations) as the retrieval result.
        """
        k = top_k or self.top_k

        if not self.graph.nodes:
            logger.warning("Graph is empty - nothing to search.")
            return GraphSearchResult()

        query_emb = self.embedder.embed([query])[0]
        label_embs = self._ensure_node_label_embeddings()

        sims = np.dot(label_embs, query_emb) / (
            np.linalg.norm(label_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argsort(sims)[-k:][::-1]
        seed_entities = {self._node_keys[i] for i in top_indices}

        expanded: set[str] = set()
        for entity in seed_entities:
            if entity in self.graph:
                ego = nx.ego_graph(self.graph, entity, radius=self.expansion_hops)
                expanded.update(ego.nodes())

        graph_context = self._build_graph_context(expanded)
        return GraphSearchResult(graph_context=graph_context)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        mode: SearchMode = SearchMode.LOCAL,
        top_k: int | None = None,
    ) -> GraphSearchResult:
        """Dispatch to the appropriate search mode with per-query caching."""
        cache_key = (query.strip(), mode.value, top_k)
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            return cached

        if mode == SearchMode.LOCAL:
            result = self.local_search(query, top_k=top_k)
        elif mode == SearchMode.GLOBAL:
            result = self.global_search(query, top_n=top_k or self.top_k)
        elif mode == SearchMode.GRAPH_ONLY:
            result = self.graph_only_search(query, top_k=top_k)
        elif mode == SearchMode.HYBRID:
            local = self.retrieve(query, mode=SearchMode.LOCAL, top_k=top_k)
            global_result = self.retrieve(query, mode=SearchMode.GLOBAL, top_k=top_k)
            result = GraphSearchResult(
                chunk_results=local.chunk_results,
                graph_context=local.graph_context,
                community_summaries=global_result.community_summaries,
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        self._retrieval_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_graph_context(self, nodes: set[str]) -> str:
        """Format a set of graph nodes + edges as a text block for the LLM."""
        lines = []
        subgraph = self.graph.subgraph(nodes)
        for u, v, data in subgraph.edges(data=True):
            u_label = self.graph.nodes[u].get("label", u)
            v_label = self.graph.nodes[v].get("label", v)
            rel = data.get("relation_type", "related_to")
            lines.append(f"{u_label} --[{rel}]--> {v_label}")
        return "\n".join(lines)

    def _build_chunk_entity_index(self) -> dict[str, set[str]]:
        """Precompute edge provenance lookups for local search."""
        chunk_to_entities: dict[str, set[str]] = defaultdict(set)
        for u, v, data in self.graph.edges(data=True):
            for chunk_id in data.get("chunk_ids", []):
                chunk_to_entities[str(chunk_id)].update((u, v))
        return chunk_to_entities

    def _ensure_summary_embeddings(self) -> None:
        """Embed community summaries once and reuse them for global search."""
        if self._summary_embeddings is not None:
            return

        self._summary_records = [
            (community.community_id, community.summary)
            for community in self.communities
            if community.summary.strip()
        ]
        if not self._summary_records:
            return

        summaries = [summary for _, summary in self._summary_records]
        self._summary_embeddings = np.asarray(self.embedder.embed(summaries), dtype=np.float32)

    def _ensure_node_label_embeddings(self) -> np.ndarray:
        """Embed graph node labels once and reuse them for graph-only search."""
        if self._node_label_embeddings is None:
            logger.info(
                "Caching graph-only node label embeddings for %d nodes.",
                len(self._node_labels),
            )
            self._node_label_embeddings = np.asarray(
                self.embedder.embed(self._node_labels),
                dtype=np.float32,
            )
        return self._node_label_embeddings
