# =============================================================================
# tests/test_metrics.py
# Tests for evaluation metric functions.
# =============================================================================
"""
Tests — Evaluation Metrics
===========================
"""

from src.evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


class TestRetrievalMetrics:
    """Test retrieval quality metrics."""

    def test_precision_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_partial(self):
        retrieved = ["a", "x", "c", "y", "z"]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=5) == 2 / 5

    def test_recall_at_k(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, k=2) == 2 / 4

    def test_mrr_first(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_third(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1 / 3

    def test_mrr_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0
