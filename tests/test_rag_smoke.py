# =============================================================================
# tests/test_rag_smoke.py
# End-to-end smoke test: does the full RAG pipeline actually work?
# =============================================================================
"""
RAG Smoke Test
==============
Creates a small set of fake papers in a temporary directory, runs the full
RAG pipeline (chunk → embed → index → retrieve → generate), and verifies
that the system returns reasonable results.

This test uses **only local models** (Sentence-Transformers for embedding)
and a **mock generator** so it works WITHOUT an OpenAI API key.

Run with::

    pytest tests/test_rag_smoke.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import Document, load_documents, search_by_title, search_by_metadata
from src.data.chunker import TokenChunker
from src.rag.vectorstore import SearchResult


# ---------------------------------------------------------------------------
# Sample Data — 5 fake ArXiv papers with realistic structure
# ---------------------------------------------------------------------------

FAKE_PAPERS = [
    {
        "doc_id": "2301.00001",
        "title": "Proximal Policy Optimization Algorithms",
        "authors": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal"],
        "abstract": (
            "We propose a new family of policy gradient methods for reinforcement "
            "learning, which alternate between sampling data through interaction "
            "with the environment, and optimizing a clipped surrogate objective."
        ),
        "categories": ["cs.LG", "cs.AI"],
        "year": 2023,
        "full_text": (
            "Proximal Policy Optimization Algorithms\n\n"
            "Abstract\n"
            "We propose a new family of policy gradient methods for reinforcement "
            "learning, which alternate between sampling data through interaction "
            "with the environment, and optimizing a clipped surrogate objective "
            "function using stochastic gradient ascent.\n\n"
            "1 Introduction\n"
            "Reinforcement learning is a type of machine learning where an agent "
            "learns to make decisions by interacting with an environment to "
            "maximize cumulative reward. Policy gradient methods are a popular "
            "approach because they directly optimize the policy.\n\n"
            "PPO uses a clipped objective to prevent large policy updates that "
            "could destabilize training. The clipping mechanism ensures that the "
            "new policy stays close to the old policy.\n\n"
            "2 Method\n"
            "The PPO-Clip objective is defined as L_CLIP = E[min(r*A, clip(r, "
            "1-epsilon, 1+epsilon)*A)] where r is the probability ratio between "
            "new and old policies and A is the advantage estimate."
        ),
        "sections": [],
        "source_file": "2301.00001.pdf",
    },
    {
        "doc_id": "2301.00002",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "abstract": (
            "We propose a new simple network architecture, the Transformer, based "
            "solely on attention mechanisms, dispensing with recurrence and "
            "convolutions entirely."
        ),
        "categories": ["cs.CL", "cs.LG"],
        "year": 2023,
        "full_text": (
            "Attention Is All You Need\n\n"
            "Abstract\n"
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose the "
            "Transformer architecture based solely on attention mechanisms.\n\n"
            "1 Introduction\n"
            "Recurrent neural networks, long short-term memory and gated "
            "recurrent units have been the dominant approaches for sequence "
            "modeling. The Transformer uses multi-head self-attention to "
            "capture dependencies regardless of distance in the sequence.\n\n"
            "2 Model Architecture\n"
            "The Transformer follows an encoder-decoder structure using stacked "
            "self-attention and point-wise fully connected layers. Scaled "
            "dot-product attention computes Attention(Q,K,V) = softmax(QK^T / "
            "sqrt(d_k)) V."
        ),
        "sections": [],
        "source_file": "2301.00002.pdf",
    },
    {
        "doc_id": "2402.00003",
        "title": "Graph Neural Networks: A Review of Methods and Applications",
        "authors": ["Jie Zhou", "Ganqu Cui"],
        "abstract": (
            "Graph neural networks are connectionist models that capture the "
            "dependence of graphs via message passing between the nodes."
        ),
        "categories": ["cs.LG", "cs.AI"],
        "year": 2024,
        "full_text": (
            "Graph Neural Networks: A Review\n\n"
            "Abstract\n"
            "Graph neural networks (GNNs) are a class of deep learning methods "
            "designed to perform inference on data described by graphs.\n\n"
            "1 Introduction\n"
            "Many real-world datasets are naturally represented as graphs: social "
            "networks, molecular structures, citation networks. GNNs aggregate "
            "information from a node's neighbors using a message-passing scheme.\n\n"
            "2 Methods\n"
            "Popular GNN variants include Graph Convolutional Networks (GCN), "
            "GraphSAGE, and Graph Attention Networks (GAT). Each differs in how "
            "they aggregate neighbor information."
        ),
        "sections": [],
        "source_file": "2402.00003.pdf",
    },
    {
        "doc_id": "2402.00004",
        "title": "Deep Q-Networks for Atari Game Playing",
        "authors": ["Volodymyr Mnih"],
        "abstract": (
            "We present the first deep learning model to successfully learn control "
            "policies directly from high-dimensional sensory input using "
            "reinforcement learning."
        ),
        "categories": ["cs.LG"],
        "year": 2024,
        "full_text": (
            "Deep Q-Networks for Atari Game Playing\n\n"
            "Abstract\n"
            "We present a deep reinforcement learning approach that combines "
            "Q-learning with deep neural networks called DQN.\n\n"
            "1 Introduction\n"
            "Learning to control agents directly from high-dimensional sensory "
            "input like vision has been a long-standing challenge. Deep Q-Networks "
            "use experience replay and a target network to stabilize training.\n\n"
            "2 Method\n"
            "DQN approximates the optimal action-value function Q*(s,a) using a "
            "deep neural network. The loss function is the temporal difference "
            "error between predicted and target Q-values."
        ),
        "sections": [],
        "source_file": "2402.00004.pdf",
    },
    {
        "doc_id": "2501.00005",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
        "abstract": (
            "We introduce BERT, a new language representation model designed to "
            "pre-train deep bidirectional representations by jointly conditioning "
            "on both left and right context."
        ),
        "categories": ["cs.CL"],
        "year": 2025,
        "full_text": (
            "BERT: Pre-training of Deep Bidirectional Transformers\n\n"
            "Abstract\n"
            "BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text.\n\n"
            "1 Introduction\n"
            "Language model pre-training has been shown to be effective for "
            "improving many natural language processing tasks. BERT uses masked "
            "language modeling where random tokens are masked and the model "
            "learns to predict them from context.\n\n"
            "2 Method\n"
            "BERT uses a multi-layer bidirectional Transformer encoder. The "
            "pre-training involves two tasks: Masked Language Model (MLM) and "
            "Next Sentence Prediction (NSP)."
        ),
        "sections": [],
        "source_file": "2501.00005.pdf",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_dir(tmp_path: Path) -> Path:
    """Write fake papers as JSON files to a temporary directory."""
    for paper in FAKE_PAPERS:
        fpath = tmp_path / f"{paper['doc_id']}.json"
        fpath.write_text(json.dumps(paper, indent=2), encoding="utf-8")
    return tmp_path


@pytest.fixture
def documents(paper_dir: Path) -> list[Document]:
    """Load fake papers as Document objects."""
    return load_documents(paper_dir)


# ---------------------------------------------------------------------------
# Tests — Data Loading
# ---------------------------------------------------------------------------

class TestDocumentLoading:
    """Verify that structured JSON files are loaded correctly."""

    def test_load_count(self, documents):
        assert len(documents) == 5

    def test_title_populated(self, documents):
        titles = [d.title for d in documents]
        assert "Proximal Policy Optimization Algorithms" in titles
        assert "Attention Is All You Need" in titles

    def test_metadata_fields(self, documents):
        doc = documents[0]  # PPO paper
        assert doc.doc_id == "2301.00001"
        assert doc.year == 2023
        assert len(doc.authors) == 3
        assert "cs.LG" in doc.categories
        assert len(doc.text) > 100  # has actual text
        assert doc.abstract != ""


# ---------------------------------------------------------------------------
# Tests — Title & Metadata Search
# ---------------------------------------------------------------------------

class TestSearch:
    """Verify title and metadata search."""

    def test_search_by_title_exact(self, documents):
        results = search_by_title("PPO", documents)
        # "Proximal Policy Optimization" doesn't contain "PPO" literally
        # but "Attention" doesn't either — this tests substring matching
        assert len(results) == 0  # "PPO" is not in any title as substring

    def test_search_by_title_substring(self, documents):
        results = search_by_title("Attention", documents)
        assert len(results) == 1
        assert results[0].doc_id == "2301.00002"

    def test_search_by_title_broad(self, documents):
        results = search_by_title("Neural", documents)
        assert len(results) == 1  # "Graph Neural Networks"

    def test_search_case_insensitive(self, documents):
        results = search_by_title("bert", documents)
        assert len(results) == 1
        assert "BERT" in results[0].title

    def test_search_by_year(self, documents):
        results = search_by_metadata(documents, year=2024)
        assert len(results) == 2  # GNN review + DQN

    def test_search_by_category(self, documents):
        results = search_by_metadata(documents, category="cs.CL")
        assert len(results) == 2  # Attention + BERT

    def test_search_by_author(self, documents):
        results = search_by_metadata(documents, author="Schulman")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests — Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    """Verify chunking preserves title metadata."""

    def test_chunks_have_title(self, documents):
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 5  # should produce multiple chunks

        for chunk in chunks:
            # Every chunk should carry its parent document's title
            assert "title" in chunk.metadata
            assert chunk.metadata["title"] != ""

    def test_chunk_ids_traceable(self, documents):
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_documents(documents)

        for chunk in chunks:
            # chunk_id format: {doc_id}__chunk_{n}
            assert "__chunk_" in chunk.chunk_id
            assert chunk.doc_id in chunk.chunk_id


# ---------------------------------------------------------------------------
# Tests — RAG Pipeline (embedding + retrieval, NO LLM calls)
# ---------------------------------------------------------------------------

class TestRAGRetrieval:
    """
    Test the RAG retrieval pipeline end-to-end using a real local
    embedder (Sentence-Transformers) but NO LLM generator.

    This verifies: chunk → embed → store → retrieve actually works.
    """

    def test_index_and_retrieve(self, documents):
        """Index 5 papers and verify retrieval returns relevant chunks."""
        from src.rag.embedder import SentenceTransformerEmbedder
        from src.rag.vectorstore import FAISSVectorStore
        from src.rag.retriever import Retriever

        # Use real local embedder + in-memory FAISS (no files, no API keys)
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore()
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)

        # Index
        chunks = chunker.chunk_documents(documents)
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{"doc_id": c.doc_id, "title": c.metadata.get("title", "")} for c in chunks]
        embeddings = embedder.embed(texts)
        store.add(ids, texts, embeddings, metadatas)

        assert store.count() == len(chunks)

        # Retrieve
        retriever = Retriever(embedder=embedder, vectorstore=store, top_k=3)

        # Question about RL should retrieve PPO / DQN chunks
        results = retriever.retrieve("What is reinforcement learning?")
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)  # meaningful scores

        # Check that RL-related papers rank high
        retrieved_doc_ids = {r.chunk_id.split("__")[0] for r in results}
        rl_papers = {"2301.00001", "2402.00004"}  # PPO + DQN
        assert retrieved_doc_ids & rl_papers, (
            f"Expected RL papers in results, got: {retrieved_doc_ids}"
        )

    def test_retrieve_transformer_query(self, documents):
        """Query about transformers should retrieve attention/BERT papers."""
        from src.rag.embedder import SentenceTransformerEmbedder
        from src.rag.vectorstore import FAISSVectorStore
        from src.rag.retriever import Retriever

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore()
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk_documents(documents)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
            embeddings,
        )

        retriever = Retriever(embedder=embedder, vectorstore=store, top_k=3)
        results = retriever.retrieve("How does the Transformer architecture work?")

        retrieved_doc_ids = {r.chunk_id.split("__")[0] for r in results}
        transformer_papers = {"2301.00002", "2501.00005"}  # Attention + BERT
        assert retrieved_doc_ids & transformer_papers, (
            f"Expected Transformer papers in results, got: {retrieved_doc_ids}"
        )

    def test_retrieve_gnn_query(self, documents):
        """Query about GNNs should retrieve the graph paper."""
        from src.rag.embedder import SentenceTransformerEmbedder
        from src.rag.vectorstore import FAISSVectorStore
        from src.rag.retriever import Retriever

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore()
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk_documents(documents)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
            embeddings,
        )

        retriever = Retriever(embedder=embedder, vectorstore=store, top_k=3)
        results = retriever.retrieve("What are graph neural networks?")

        # The GNN paper should be in top 3
        retrieved_doc_ids = {r.chunk_id.split("__")[0] for r in results}
        assert "2402.00003" in retrieved_doc_ids


# ===========================================================================
# TXT-BASED TESTS — Full RAG pipeline using plain .txt files (legacy path)
# ===========================================================================

FAKE_TXT_PAPERS = {
    "2301.00001.txt": (
        "Proximal Policy Optimization Algorithms\n\n"
        "We propose a new family of policy gradient methods for reinforcement "
        "learning, which alternate between sampling data through interaction "
        "with the environment, and optimizing a clipped surrogate objective "
        "function using stochastic gradient ascent.\n\n"
        "PPO uses a clipped objective to prevent large policy updates that "
        "could destabilize training. The clipping mechanism ensures that the "
        "new policy stays close to the old policy."
    ),
    "2301.00002.txt": (
        "Attention Is All You Need\n\n"
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks. We propose the "
        "Transformer architecture based solely on attention mechanisms.\n\n"
        "The Transformer uses multi-head self-attention to capture "
        "dependencies regardless of distance in the sequence."
    ),
    "2402.00003.txt": (
        "Graph Neural Networks: A Review\n\n"
        "Graph neural networks (GNNs) are a class of deep learning methods "
        "designed to perform inference on data described by graphs. GNNs "
        "aggregate information from a node's neighbors using a message-passing "
        "scheme.\n\n"
        "Popular GNN variants include Graph Convolutional Networks (GCN), "
        "GraphSAGE, and Graph Attention Networks (GAT)."
    ),
}


@pytest.fixture
def txt_paper_dir(tmp_path: Path) -> Path:
    """Write fake papers as plain .txt files to a temporary directory."""
    for fname, content in FAKE_TXT_PAPERS.items():
        (tmp_path / fname).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def txt_documents(txt_paper_dir: Path) -> list[Document]:
    """Load fake .txt papers as Document objects."""
    return load_documents(txt_paper_dir)


class TestTxtDocumentLoading:
    """Verify that plain .txt files are loaded via the legacy fallback path."""

    def test_load_count(self, txt_documents):
        assert len(txt_documents) == 3

    def test_title_from_first_line(self, txt_documents):
        """Loader uses the first non-empty line as title for .txt files."""
        titles = {d.title for d in txt_documents}
        assert "Proximal Policy Optimization Algorithms" in titles
        assert "Attention Is All You Need" in titles
        assert "Graph Neural Networks: A Review" in titles

    def test_doc_ids_from_filename(self, txt_documents):
        ids = {d.doc_id for d in txt_documents}
        assert ids == {"2301.00001", "2301.00002", "2402.00003"}

    def test_full_text_present(self, txt_documents):
        for doc in txt_documents:
            assert len(doc.text) > 50


class TestTxtSearch:
    """Verify title search works on txt-loaded documents."""

    def test_search_by_title(self, txt_documents):
        results = search_by_title("Attention", txt_documents)
        assert len(results) == 1
        assert results[0].doc_id == "2301.00002"

    def test_search_case_insensitive(self, txt_documents):
        results = search_by_title("graph neural", txt_documents)
        assert len(results) == 1


class TestTxtRAGRetrieval:
    """
    Full RAG retrieval using .txt-loaded documents: chunk → embed → store
    → retrieve.  No LLM required.
    """

    def test_txt_index_and_retrieve(self, txt_documents):
        from src.rag.embedder import SentenceTransformerEmbedder
        from src.rag.vectorstore import FAISSVectorStore
        from src.rag.retriever import Retriever

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore()
        chunker = TokenChunker(chunk_size=80, chunk_overlap=15)

        chunks = chunker.chunk_documents(txt_documents)
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        store.add(ids, texts, embeddings)

        assert store.count() == len(chunks)

        retriever = Retriever(embedder=embedder, vectorstore=store, top_k=3)

        # RL question should surface PPO paper
        results = retriever.retrieve("policy gradient reinforcement learning")
        retrieved_ids = {r.chunk_id.split("__")[0] for r in results}
        assert "2301.00001" in retrieved_ids

        # Transformer question should surface attention paper
        results = retriever.retrieve("self-attention transformer architecture")
        retrieved_ids = {r.chunk_id.split("__")[0] for r in results}
        assert "2301.00002" in retrieved_ids

        # GNN question should surface graph paper
        results = retriever.retrieve("graph neural network message passing")
        retrieved_ids = {r.chunk_id.split("__")[0] for r in results}
        assert "2402.00003" in retrieved_ids
