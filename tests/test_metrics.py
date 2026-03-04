# =============================================================================
# tests/test_metrics.py
# Tests for evaluation metric functions.
# =============================================================================
"""
Tests — Evaluation Metrics
===========================
"""

from src.evaluation.metrics import (
    _parse_faithfulness_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


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


class TestFaithfulnessParsing:
    """Test parsing for LLM-as-judge numeric responses."""

    def test_parse_plain_number(self):
        assert _parse_faithfulness_score("0.83") == 0.83

    def test_parse_number_with_text(self):
        assert _parse_faithfulness_score("Score: 0.6") == 0.6

    def test_parse_clamps_high_values(self):
        assert _parse_faithfulness_score("1.9") == 1.0

    def test_parse_invalid_returns_zero(self):
        assert _parse_faithfulness_score("not a number") == 0.0
