# =============================================================================
# tests/test_graph_retriever.py
# Tests for GraphRetriever search modes.
# =============================================================================

from __future__ import annotations

import numpy as np

from src.graph_rag.retriever import GraphRetriever, SearchMode
from src.rag.vectorstore import SearchResult


class DummyEmbedder:
    """Deterministic embedder for tests."""

    def embed(self, texts):
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

    def query(self, query_embedding, top_k=5):  # noqa: ARG002
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
