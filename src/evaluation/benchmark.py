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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    comprehensiveness_score,
    diversity_score,
    directness_score,
    empowerment_score,
    faithfulness_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    rouge_scores,
)

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Best-effort token estimate for benchmark reporting."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


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
    pipeline: str  # "rag", "graphrag_local", "graphrag_global", "graphrag_graph_only", etc.
    run_id: str = ""
    timestamp_utc: str = ""
    search_mode: str = ""  # search mode used (for GraphRAG)
    llm_provider: str = ""
    llm_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    top_k: int = 5
    answer: str = ""
    latency_s: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0
    mrr: float = 0.0
    rouge1: float = 0.0
    rougeL: float = 0.0
    faithfulness: float = 0.0
    comprehensiveness: float = 0.0
    diversity: float = 0.0
    directness: float = 0.0
    empowerment: float = 0.0
    context_char_count: int = 0
    estimated_context_tokens: int = 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Orchestrates side-by-side evaluation.

    Evaluates:
    - Standard RAG (vector retrieval only)
    - GraphRAG local (vector + graph expansion)
    - GraphRAG global (community summaries)
    - GraphRAG graph_only (pure graph, NO vector store)
    - GraphRAG hybrid (local + global combined)

    Parameters
    ----------
    rag_pipeline : optional
        An initialised RAGPipeline instance.
    graphrag_pipeline : optional
        An initialised GraphRAGPipeline instance.
    graphrag_modes : list[str], optional
        Which GraphRAG search modes to benchmark.  Defaults to all four.
    compute_faithfulness : bool
        Whether to run LLM-as-judge faithfulness scoring (adds latency).
    compute_quality_judges : bool
        Whether to run additional LLM judges: comprehensiveness,
        diversity, directness, and empowerment.
    """

    def __init__(
        self,
        rag_pipeline=None,
        graphrag_pipeline=None,
        graphrag_modes: list[str] | None = None,
        compute_faithfulness: bool = False,
        compute_quality_judges: bool = False,
    ):
        self.rag = rag_pipeline
        self.graphrag = graphrag_pipeline
        self.graphrag_modes = graphrag_modes or ["local", "global", "graph_only", "hybrid"]
        self.compute_faithfulness = compute_faithfulness
        self.compute_quality_judges = compute_quality_judges
        self._run_id = ""
        self._timestamp_utc = ""

        valid_modes = {"local", "global", "graph_only", "hybrid"}
        unknown_modes = set(self.graphrag_modes) - valid_modes
        if unknown_modes:
            raise ValueError(f"Unknown GraphRAG benchmark modes: {sorted(unknown_modes)}")

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

    def run(
        self,
        questions_path: Path | str,
        top_k: int = 5,
        write_jsonl_log: bool = True,
        jsonl_log_path: Path | str | None = None,
    ) -> pd.DataFrame:
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
            One row per (question, pipeline, mode) combination.
        """
        self._run_id = str(uuid.uuid4())
        self._timestamp_utc = datetime.now(timezone.utc).isoformat()
        logger.info("Benchmark run started: run_id=%s", self._run_id)

        questions = self.load_questions(questions_path)
        records: list[dict] = []

        for eq in questions:
            # --- Standard RAG ---
            if self.rag:
                rec = self._evaluate_rag(eq, top_k)
                records.append(rec.__dict__)

            # --- Graph RAG (each search mode) ---
            if self.graphrag:
                for mode in self.graphrag_modes:
                    rec = self._evaluate_graphrag(eq, top_k, mode)
                    records.append(rec.__dict__)

        df = pd.DataFrame(records)
        if write_jsonl_log:
            self._write_jsonl_log(df, jsonl_log_path)
        logger.info("Benchmark complete — %d records.", len(df))
        return df

    def _write_jsonl_log(self, df: pd.DataFrame, jsonl_log_path: Path | str | None = None) -> None:
        """Persist benchmark records as JSONL for auditability and replay."""
        if df.empty:
            return

        if jsonl_log_path is None:
            from src.config import get_settings

            base = get_settings().results_dir / "benchmark_runs"
            base.mkdir(parents=True, exist_ok=True)
            log_path = base / f"{self._run_id}.jsonl"
        else:
            log_path = Path(jsonl_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Benchmark JSONL log written: %s", log_path)

    def _get_model_metadata(self) -> dict[str, str]:
        """Collect provider/model metadata for benchmark traceability."""
        from src.config import get_settings

        settings = get_settings()

        llm_model = ""
        if self.rag is not None and getattr(self.rag, "generator", None) is not None:
            llm_model = getattr(self.rag.generator, "model", "")
        elif self.graphrag is not None and getattr(self.graphrag, "generator", None) is not None:
            llm_model = getattr(self.graphrag.generator, "model", "")

        embedding_model = ""
        if self.rag is not None and getattr(self.rag, "embedder", None) is not None:
            embedding_model = (
                getattr(self.rag.embedder, "model_name", "")
                or getattr(self.rag.embedder, "model", "")
            )
        elif self.graphrag is not None and getattr(self.graphrag, "embedder", None) is not None:
            embedding_model = (
                getattr(self.graphrag.embedder, "model_name", "")
                or getattr(self.graphrag.embedder, "model", "")
            )

        return {
            "llm_provider": settings.llm_provider.value,
            "llm_model": llm_model,
            "embedding_provider": settings.embedding_provider.value,
            "embedding_model": embedding_model,
        }

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

        context = "\n\n".join(c.text for c in result.retrieved_chunks)
        meta = self._get_model_metadata()

        rec = EvalRecord(
            run_id=self._run_id,
            timestamp_utc=self._timestamp_utc,
            question=eq.question,
            pipeline="rag",
            search_mode="vector",
            llm_provider=meta["llm_provider"],
            llm_model=meta["llm_model"],
            embedding_provider=meta["embedding_provider"],
            embedding_model=meta["embedding_model"],
            top_k=top_k,
            answer=result.answer,
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
            context_char_count=len(context),
            estimated_context_tokens=_estimate_tokens(context),
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        if self.compute_faithfulness and context.strip():
            rec.faithfulness = faithfulness_score(result.answer, context)
        if self.compute_quality_judges and context.strip():
            rec.comprehensiveness = comprehensiveness_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )
            rec.diversity = diversity_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )
            rec.directness = directness_score(
                question=eq.question,
                answer=result.answer,
            )
            rec.empowerment = empowerment_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )

        return rec

    def _evaluate_graphrag(self, eq: EvalQuestion, top_k: int, mode: str) -> EvalRecord:
        """Evaluate GraphRAG on one question with a specific search mode."""
        from src.graph_rag.retriever import SearchMode

        search_mode = SearchMode(mode)
        t0 = time.perf_counter()
        result = self.graphrag.query(eq.question, mode=search_mode, top_k=top_k)
        latency = time.perf_counter() - t0

        chunk_results = result.search_result.chunk_results if result.search_result else []
        retrieved_ids = [c.chunk_id for c in chunk_results]
        relevant = set(eq.relevant_doc_ids)

        context_parts = [c.text for c in chunk_results]
        if result.search_result and result.search_result.graph_context:
            context_parts.append(result.search_result.graph_context)
        for summary in (result.search_result.community_summaries if result.search_result else []):
            context_parts.append(summary)
        context = "\n\n".join(context_parts)
        meta = self._get_model_metadata()

        rec = EvalRecord(
            run_id=self._run_id,
            timestamp_utc=self._timestamp_utc,
            question=eq.question,
            pipeline=f"graphrag_{mode}",
            search_mode=mode,
            llm_provider=meta["llm_provider"],
            llm_model=meta["llm_model"],
            embedding_provider=meta["embedding_provider"],
            embedding_model=meta["embedding_model"],
            top_k=top_k,
            answer=result.answer,
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
            context_char_count=len(context),
            estimated_context_tokens=_estimate_tokens(context),
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        if self.compute_faithfulness and context.strip():
            rec.faithfulness = faithfulness_score(result.answer, context)
        if self.compute_quality_judges and context.strip():
            rec.comprehensiveness = comprehensiveness_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )
            rec.diversity = diversity_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )
            rec.directness = directness_score(
                question=eq.question,
                answer=result.answer,
            )
            rec.empowerment = empowerment_score(
                question=eq.question,
                answer=result.answer,
                context=context,
            )

        return rec
