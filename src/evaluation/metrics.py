# =============================================================================
# src/evaluation/metrics.py
# Retrieval and generation quality metrics.
# =============================================================================
"""
Metrics
=======
A collection of evaluation metrics used to compare RAG and GraphRAG outputs.

Retrieval Metrics
-----------------
- Precision@k, Recall@k (require ground-truth relevant doc IDs)
- Mean Reciprocal Rank (MRR)

Generation Metrics
------------------
- ROUGE-1 / ROUGE-L (lexical overlap with reference answers)
- Faithfulness score (LLM-as-judge — is the answer grounded in context?)

All functions accept simple Python types so they're easy to use from
notebooks and scripts.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """
    Precision@k — fraction of retrieved items (top-k) that are relevant.

    Parameters
    ----------
    retrieved_ids : Sequence[str]
        Ordered list of retrieved chunk/document IDs.
    relevant_ids : set[str]
        Ground-truth relevant IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Value in [0, 1].
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k — fraction of relevant items found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """MRR — reciprocal of the rank of the first relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Generation Metrics
# ---------------------------------------------------------------------------

def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1 and ROUGE-L F1 scores.

    Returns
    -------
    dict
        {"rouge1": float, "rougeL": float}
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def faithfulness_score(
    answer: str,
    context: str,
    model: str = "gpt-4o-mini",
) -> float:
    """
    LLM-as-judge: rate how faithfully the *answer* is grounded in the
    *context* on a scale of 0–1.

    Parameters
    ----------
    answer : str
        The generated answer.
    context : str
        The retrieval context provided to the generator.
    model : str
        LLM to use as judge.

    Returns
    -------
    float
        Score in [0, 1].

    TODO
    ----
    - [ ] Implement the actual LLM call.
    - [ ] Consider using a cheaper / faster judge model.
    """
    logger.warning("faithfulness_score is a stub — returning 0.0")
    # TODO: Implement via structured LLM prompt
    return 0.0
