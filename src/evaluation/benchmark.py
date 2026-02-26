# =============================================================================
# src/evaluation/benchmark.py
# Side-by-side benchmark runner for RAG vs GraphRAG.
# =============================================================================
"""
Benchmark Runner
================
Runs the same set of evaluation questions through both standard RAG and
GraphRAG pipelines, records metrics, and exports a comparison report.

Usage
-----
::

    from src.evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner()
    report = runner.run("eval_questions.json")
    report.to_csv("results/benchmark_report.csv")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    faithfulness_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    rouge_scores,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    """A single evaluation question with optional ground truth."""
    question: str
    reference_answer: str = ""
    relevant_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvalRecord:
    """Metrics for one question + one pipeline."""
    question: str
    pipeline: str  # "rag" or "graphrag"
    answer: str = ""
    latency_s: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0
    mrr: float = 0.0
    rouge1: float = 0.0
    rougeL: float = 0.0
    faithfulness: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Orchestrates side-by-side evaluation.

    Parameters
    ----------
    rag_pipeline : optional
        An initialised RAGPipeline instance.
    graphrag_pipeline : optional
        An initialised GraphRAGPipeline instance.
    """

    def __init__(self, rag_pipeline=None, graphrag_pipeline=None):
        self.rag = rag_pipeline
        self.graphrag = graphrag_pipeline

    def load_questions(self, path: Path | str) -> list[EvalQuestion]:
        """
        Load evaluation questions from a JSON file.

        Expected format::

            [
              {
                "question": "What is PPO?",
                "reference_answer": "Proximal Policy Optimization is ...",
                "relevant_doc_ids": ["2301.00001", "2301.00042"]
              },
              ...
            ]
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuestion(**item) for item in data]

    def run(self, questions_path: Path | str, top_k: int = 5) -> pd.DataFrame:
        """
        Run both pipelines on all questions and return a DataFrame of metrics.

        Parameters
        ----------
        questions_path : Path
            Path to the evaluation JSON file.
        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        pd.DataFrame
            One row per (question, pipeline) pair.
        """
        questions = self.load_questions(questions_path)
        records: list[dict] = []

        for eq in questions:
            # --- Standard RAG ---
            if self.rag:
                rec = self._evaluate_rag(eq, top_k)
                records.append(rec.__dict__)

            # --- Graph RAG ---
            if self.graphrag:
                rec = self._evaluate_graphrag(eq, top_k)
                records.append(rec.__dict__)

        df = pd.DataFrame(records)
        logger.info("Benchmark complete — %d records.", len(df))
        return df

    # ------------------------------------------------------------------
    # Private evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_rag(self, eq: EvalQuestion, top_k: int) -> EvalRecord:
        """Evaluate standard RAG on one question."""
        t0 = time.perf_counter()
        result = self.rag.query(eq.question, top_k=top_k)
        latency = time.perf_counter() - t0

        retrieved_ids = [c.chunk_id for c in result.retrieved_chunks]
        relevant = set(eq.relevant_doc_ids)

        rec = EvalRecord(
            question=eq.question,
            pipeline="rag",
            answer=result.answer,
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        return rec

    def _evaluate_graphrag(self, eq: EvalQuestion, top_k: int) -> EvalRecord:
        """Evaluate GraphRAG on one question."""
        t0 = time.perf_counter()
        result = self.graphrag.query(eq.question, top_k=top_k)
        latency = time.perf_counter() - t0

        chunk_results = result.search_result.chunk_results if result.search_result else []
        retrieved_ids = [c.chunk_id for c in chunk_results]
        relevant = set(eq.relevant_doc_ids)

        rec = EvalRecord(
            question=eq.question,
            pipeline="graphrag",
            answer=result.answer,
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        return rec
