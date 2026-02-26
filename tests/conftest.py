# =============================================================================
# tests/conftest.py
# Shared pytest fixtures for the test suite.
# =============================================================================
"""
Test Configuration
==================
Provides reusable fixtures: sample documents, chunks, mock embedders, etc.
"""

import pytest

from src.data.loader import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return a small set of fake documents for unit testing."""
    return [
        Document(
            doc_id="test_001",
            title="Introduction to Reinforcement Learning",
            abstract="A survey of RL algorithms including PPO.",
            authors=["Alice Smith", "Bob Jones"],
            categories=["cs.LG"],
            year=2023,
            text=(
                "Reinforcement learning is a type of machine learning where an agent "
                "learns to make decisions by interacting with an environment. "
                "Proximal Policy Optimization (PPO) is a popular RL algorithm."
            ),
            metadata={},
        ),
        Document(
            doc_id="test_002",
            title="A Survey on Graph Neural Networks",
            abstract="GNNs operate on graph-structured data.",
            authors=["Carol Lee"],
            categories=["cs.LG", "cs.AI"],
            year=2024,
            text=(
                "Graph Neural Networks (GNNs) operate on graph-structured data. "
                "They aggregate information from neighbouring nodes to learn "
                "node, edge, or graph-level representations."
            ),
            metadata={},
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Pre-built chunks for testing retrieval without running the chunker."""
    from src.data.chunker import Chunk

    return [
        Chunk(chunk_id="test_001__chunk_0", doc_id="test_001",
              text="Reinforcement learning is a type of machine learning.", index=0),
        Chunk(chunk_id="test_001__chunk_1", doc_id="test_001",
              text="PPO is a popular RL algorithm.", index=1),
        Chunk(chunk_id="test_002__chunk_0", doc_id="test_002",
              text="Graph Neural Networks operate on graph-structured data.", index=0),
    ]
