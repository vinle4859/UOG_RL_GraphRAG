# =============================================================================
# tests/test_graph_retriever.py
# Tests for GraphRetriever search modes.
# =============================================================================

from __future__ import annotations

import numpy as np

from src.graph_rag.community import Community
from src.graph_rag.retriever import GraphRetriever, SearchMode
from src.rag.vectorstore import SearchResult


class DummyEmbedder:
    """Deterministic embedder for tests."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            t = text.lower()
            vectors.append(
                [
                    1.0 if "ppo" in t else 0.0,
                    1.0 if "reinforcement" in t else 0.0,
                    1.0 if "transformer" in t else 0.0,
                ]
            )
        return np.array(vectors, dtype=np.float32)


class DummyVectorStore:
    """Simple in-memory vector store stub for local mode tests."""

    def __init__(self):
        self.query_calls = 0

    def query(self, query_embedding, top_k=5):  # noqa: ARG002
        self.query_calls += 1
        return [
            SearchResult(
                chunk_id="doc1__chunk_0",
                text="PPO is a policy gradient algorithm.",
                score=0.9,
                metadata={"doc_id": "doc1"},
            )
        ]


def _build_graph():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("ppo", label="PPO", entity_type="Algorithm")
    graph.add_node("reinforcement learning", label="Reinforcement Learning", entity_type="Concept")
    graph.add_edge(
        "ppo",
        "reinforcement learning",
        relation_type="is_part_of",
        chunk_ids=["doc1__chunk_0"],
    )
    return graph


class TestGraphRetriever:
    def test_graph_only_mode_returns_graph_context(self):
        graph = _build_graph()
        retriever = GraphRetriever(
            graph=graph,
            communities=[],
            embedder=DummyEmbedder(),
            vectorstore=DummyVectorStore(),
            top_k=1,
        )

        result = retriever.retrieve("What is PPO?", mode=SearchMode.GRAPH_ONLY, top_k=1)
        assert result.graph_context
        assert "PPO --[is_part_of]--> Reinforcement Learning" in result.graph_context

    def test_local_mode_returns_chunk_and_graph_context(self):
        graph = _build_graph()
        retriever = GraphRetriever(
            graph=graph,
            communities=[],
            embedder=DummyEmbedder(),
            vectorstore=DummyVectorStore(),
            top_k=1,
        )

        result = retriever.retrieve("What is PPO?", mode=SearchMode.LOCAL, top_k=1)
        assert len(result.chunk_results) == 1
        assert result.chunk_results[0].chunk_id == "doc1__chunk_0"
        assert "PPO --[is_part_of]--> Reinforcement Learning" in result.graph_context

    def test_graph_only_mode_caches_repeated_queries(self):
        graph = _build_graph()
        embedder = DummyEmbedder()
        retriever = GraphRetriever(
            graph=graph,
            communities=[],
            embedder=embedder,
            vectorstore=DummyVectorStore(),
            top_k=1,
        )

        retriever.retrieve("What is PPO?", mode=SearchMode.GRAPH_ONLY, top_k=1)
        calls_after_first = len(embedder.calls)
        retriever.retrieve("What is PPO?", mode=SearchMode.GRAPH_ONLY, top_k=1)

        assert calls_after_first == 2
        assert len(embedder.calls) == calls_after_first

    def test_hybrid_mode_reuses_cached_local_and_global_results(self):
        graph = _build_graph()
        embedder = DummyEmbedder()
        vectorstore = DummyVectorStore()
        retriever = GraphRetriever(
            graph=graph,
            communities=[
                Community(
                    community_id=1, nodes=["ppo"], summary="PPO belongs to reinforcement learning."
                )
            ],
            embedder=embedder,
            vectorstore=vectorstore,
            top_k=1,
        )

        retriever.retrieve("What is PPO?", mode=SearchMode.LOCAL, top_k=1)
        retriever.retrieve("What is PPO?", mode=SearchMode.GLOBAL, top_k=1)
        embed_calls_before_hybrid = len(embedder.calls)
        vector_queries_before_hybrid = vectorstore.query_calls

        result = retriever.retrieve("What is PPO?", mode=SearchMode.HYBRID, top_k=1)

        assert result.chunk_results
        assert result.community_summaries
        assert len(embedder.calls) == embed_calls_before_hybrid
        assert vectorstore.query_calls == vector_queries_before_hybrid
