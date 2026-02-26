# =============================================================================
# tests/test_data.py
# Tests for the data loading and chunking modules.
# =============================================================================
"""
Tests — Data Pipeline
=====================
"""

from src.data.chunker import TokenChunker
from src.data.loader import Document


class TestDocument:
    """Test the Document dataclass."""

    def test_creation(self):
        doc = Document(doc_id="abc", text="Hello world")
        assert doc.doc_id == "abc"
        assert doc.text == "Hello world"
        assert doc.metadata == {}


class TestTokenChunker:
    """Test the token-based chunker."""

    def test_single_chunk(self):
        """Short documents should produce exactly one chunk."""
        chunker = TokenChunker(chunk_size=512, chunk_overlap=64)
        doc = Document(doc_id="short", text="A short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].doc_id == "short"
        assert chunks[0].index == 0

    def test_multiple_chunks(self, sample_documents):
        """Documents longer than chunk_size should produce >1 chunk."""
        chunker = TokenChunker(chunk_size=10, chunk_overlap=2)
        for doc in sample_documents:
            chunks = chunker.chunk_document(doc)
            assert len(chunks) >= 1
            for i, chunk in enumerate(chunks):
                assert chunk.index == i
                assert chunk.doc_id == doc.doc_id

    def test_chunk_ids_unique(self, sample_documents):
        """All chunk IDs should be unique across documents."""
        chunker = TokenChunker(chunk_size=20, chunk_overlap=4)
        all_chunks = chunker.chunk_documents(sample_documents)
        ids = [c.chunk_id for c in all_chunks]
        assert len(ids) == len(set(ids))
