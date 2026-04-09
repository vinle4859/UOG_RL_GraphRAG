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
from typing import Any

import pandas as pd

from src.evaluation.metrics import (
    assess_comprehensiveness,
    assess_directness,
    assess_diversity,
    assess_empowerment,
    assess_faithfulness,
    comprehensiveness_score,
    directness_score,
    diversity_score,
    empowerment_score,
    faithfulness_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    rouge_scores,
)

logger = logging.getLogger(__name__)

_SCOPE_VALUES = {"local", "global"}
_UNIT_METRICS = (
    "precision_5",
    "recall_5",
    "mrr",
    "rouge1",
    "rougeL",
    "faithfulness",
    "comprehensiveness",
    "diversity",
    "directness",
    "empowerment",
)
_RAG_MAX_CHARS = 500


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


def _normalise_question_scope(raw_scope: Any) -> str:
    """Return a supported question scope label."""
    scope = str(raw_scope or "local").strip().lower()
    if scope in _SCOPE_VALUES:
        return scope
    logger.warning("Unknown question_scope '%s' in eval set; defaulting to 'local'.", raw_scope)
    return "local"


def _question_from_payload(item: dict[str, Any]) -> EvalQuestion:
    """Build EvalQuestion with backwards-compatible parsing."""
    if not isinstance(item, dict):
        raise ValueError("Each eval question must be a JSON object.")

    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError("Each eval question requires a non-empty 'question' field.")

    relevant = item.get("relevant_doc_ids", [])
    if not isinstance(relevant, list):
        relevant = []

    return EvalQuestion(
        question=question,
        reference_answer=str(item.get("reference_answer", "")).strip(),
        relevant_doc_ids=[str(x).strip() for x in relevant if str(x).strip()],
        question_scope=_normalise_question_scope(item.get("question_scope", "local")),
    )


def _safe_metric(value: float, metric_name: str) -> float:
    """Clamp benchmark metrics to [0, 1] while preserving run continuity."""
    if value < 0.0 or value > 1.0:
        logger.warning("Metric %s out of bounds (%.4f); clamping to [0, 1].", metric_name, value)
    return min(1.0, max(0.0, float(value)))


def _validate_record(rec: EvalRecord) -> EvalRecord:
    """Validate and normalise benchmark record fields before persistence."""
    rec.question_scope = _normalise_question_scope(rec.question_scope)
    rec.top_k = max(1, int(rec.top_k))
    rec.relevant_doc_count = max(0, int(rec.relevant_doc_count))
    rec.retrieved_count = max(0, int(rec.retrieved_count))
    rec.relevant_hits_at_k = max(0, int(rec.relevant_hits_at_k))
    rec.has_relevant_hit_at_k = bool(rec.relevant_hits_at_k > 0)
    rec.latency_s = max(0.0, float(rec.latency_s))
    rec.context_char_count = max(0, int(rec.context_char_count))
    rec.estimated_context_tokens = max(0, int(rec.estimated_context_tokens))

    for metric_name in _UNIT_METRICS:
        setattr(rec, metric_name, _safe_metric(float(getattr(rec, metric_name)), metric_name))

    return rec


def _doc_id_from_chunk_id(chunk_id: str) -> str:
    """Extract doc id prefix from chunk id values like 'doc__chunk_0'."""
    return chunk_id.split("__", 1)[0] if "__" in chunk_id else chunk_id


def _safe_json_list(raw: Any) -> list[Any]:
    """Parse JSON-encoded lists in benchmark rows; tolerate malformed input."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _format_chunk_pairs(chunk_ids: list[str], scores: list[Any], limit: int = 5) -> str:
    """Return compact human-readable top-k chunk lines."""
    out: list[str] = []
    for idx, cid in enumerate(chunk_ids[:limit]):
        score_txt = "n/a"
        if idx < len(scores):
            try:
                score_txt = f"{float(scores[idx]):.4f}"
            except Exception:
                score_txt = str(scores[idx])
        out.append(f"{idx + 1}. {cid} (score={score_txt})")
    return "\n".join(out)


def _build_chunk_payload(
    chunk_results: list[Any], limit: int | None = None
) -> list[dict[str, Any]]:
    """Serialise retrieved chunks for downstream human review."""
    payload: list[dict[str, Any]] = []
    items = chunk_results if limit is None else chunk_results[:limit]
    for idx, c in enumerate(items, start=1):
        payload.append(
            {
                "rank": idx,
                "chunk_id": getattr(c, "chunk_id", ""),
                "score": round(float(getattr(c, "score", 0.0)), 6),
                "text": getattr(c, "text", "") or "",
                "metadata": getattr(c, "metadata", None) or {},
            }
        )
    return payload


def _format_chunk_details(payload: list[dict[str, Any]], text_limit: int = _RAG_MAX_CHARS) -> str:
    """Pretty-print top-k chunk details with text and metadata."""
    if not payload:
        return "(none)"

    lines: list[str] = []
    for item in payload:
        text = str(item.get("text", "") or "").strip().replace("\n", " ")
        if len(text) > text_limit:
            text = text[:text_limit] + "..."
        meta = item.get("metadata", {})
        meta_txt = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"{item.get('rank', '?')}. {item.get('chunk_id', '')} "
            f"(score={float(item.get('score', 0.0)):.4f})\n"
            f"metadata: {meta_txt}\n"
            f"text: {text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    """A single evaluation question with optional ground truth."""

    question: str
    reference_answer: str = ""
    relevant_doc_ids: list[str] = field(default_factory=list)
    question_scope: str = "local"  # "local" or "global"


@dataclass
class EvalRecord:
    """Metrics for one question + one pipeline."""

    question: str
    pipeline: str  # "rag", "graphrag_local", "graphrag_global", "graphrag_graph_only", etc.
    question_scope: str = "local"
    run_id: str = ""
    timestamp_utc: str = ""
    search_mode: str = ""  # search mode used (for GraphRAG)
    llm_provider: str = ""
    llm_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    top_k: int = 5
    answer: str = ""
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    latency_s: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0
    mrr: float = 0.0
    rouge1: float = 0.0
    rougeL: float = 0.0  # noqa: N815 - kept for backward-compatible report schema
    faithfulness: float = 0.0
    comprehensiveness: float = 0.0
    diversity: float = 0.0
    directness: float = 0.0
    empowerment: float = 0.0
    relevant_doc_count: int = 0
    retrieved_count: int = 0
    relevant_hits_at_k: int = 0
    has_relevant_hit_at_k: bool = False
    retrieved_chunk_ids: str = ""  # JSON-encoded list
    retrieved_chunk_ids_top5: str = ""  # JSON-encoded list
    retrieved_chunk_scores: str = ""  # JSON-encoded list[float]
    retrieved_doc_ids: str = ""  # JSON-encoded unique list
    retrieved_chunk_payload: str = ""  # JSON-encoded list of chunk details
    retrieved_chunk_payload_top5: str = ""  # JSON-encoded list of top-5 chunk details
    faithfulness_rationale: str = ""
    comprehensiveness_rationale: str = ""
    diversity_rationale: str = ""
    directness_rationale: str = ""
    empowerment_rationale: str = ""
    judge_validation_notes: str = ""
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
    compute_efficiency_tracking : bool
        Whether to compute efficiency metrics (latency/context size/token estimate).
    """

    def __init__(
        self,
        rag_pipeline=None,
        graphrag_pipeline=None,
        graphrag_modes: list[str] | None = None,
        compute_faithfulness: bool = True,
        compute_quality_judges: bool = True,
        compute_efficiency_tracking: bool = True,
        collect_judge_explanations: bool = True,
    ):
        self.rag = rag_pipeline
        self.graphrag = graphrag_pipeline
        self.graphrag_modes = graphrag_modes or ["local", "global", "graph_only", "hybrid"]
        self.compute_faithfulness = compute_faithfulness
        self.compute_quality_judges = compute_quality_judges
        self.compute_efficiency_tracking = compute_efficiency_tracking
        self.collect_judge_explanations = collect_judge_explanations
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

        if not isinstance(data, list):
            raise ValueError("Evaluation question file must contain a JSON array.")

        questions: list[EvalQuestion] = []
        for item in data:
            questions.append(_question_from_payload(item))

        logger.info("Loaded %d eval questions from %s", len(questions), path)
        return questions

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

    def build_human_review_df(self, df: pd.DataFrame, chunk_limit: int = 5) -> pd.DataFrame:
        """Create a compact, readable table for manual benchmark inspection."""
        if df.empty:
            return df.copy()

        rows: list[dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            chunk_ids = _safe_json_list(rec.get("retrieved_chunk_ids", ""))
            chunk_scores = _safe_json_list(rec.get("retrieved_chunk_scores", ""))
            chunk_payload_top5 = _safe_json_list(rec.get("retrieved_chunk_payload_top5", ""))
            rows.append(
                {
                    "question_scope": rec.get("question_scope", "local"),
                    "question": rec.get("question", ""),
                    "pipeline": rec.get("pipeline", ""),
                    "search_mode": rec.get("search_mode", ""),
                    "answer": rec.get("answer", ""),
                    "top_k_chunks": _format_chunk_pairs(chunk_ids, chunk_scores, limit=chunk_limit),
                    "top_k_chunk_details": _format_chunk_details(
                        chunk_payload_top5, text_limit=_RAG_MAX_CHARS
                    ),
                    "retrieved_count": rec.get("retrieved_count", 0),
                    "relevant_hits_at_k": rec.get("relevant_hits_at_k", 0),
                    "precision_5": rec.get("precision_5", 0.0),
                    "recall_5": rec.get("recall_5", 0.0),
                    "mrr": rec.get("mrr", 0.0),
                    "faithfulness": rec.get("faithfulness", 0.0),
                    "comprehensiveness": rec.get("comprehensiveness", 0.0),
                    "diversity": rec.get("diversity", 0.0),
                    "directness": rec.get("directness", 0.0),
                    "empowerment": rec.get("empowerment", 0.0),
                    "faithfulness_rationale": rec.get("faithfulness_rationale", ""),
                    "comprehensiveness_rationale": rec.get("comprehensiveness_rationale", ""),
                    "diversity_rationale": rec.get("diversity_rationale", ""),
                    "directness_rationale": rec.get("directness_rationale", ""),
                    "empowerment_rationale": rec.get("empowerment_rationale", ""),
                    "judge_validation_notes": rec.get("judge_validation_notes", ""),
                    "latency_s": rec.get("latency_s", 0.0),
                    "estimated_context_tokens": rec.get("estimated_context_tokens", 0),
                }
            )

        review_df = pd.DataFrame(rows)
        sort_cols = [
            c for c in ["question_scope", "question", "pipeline"] if c in review_df.columns
        ]
        if sort_cols:
            review_df = review_df.sort_values(sort_cols).reset_index(drop=True)
        return review_df

    def write_human_review_markdown(
        self,
        df: pd.DataFrame,
        output_path: Path | str,
        chunk_limit: int = 5,
    ) -> None:
        """Write a markdown report optimised for human-in-the-loop review."""
        review_df = self.build_human_review_df(df, chunk_limit=chunk_limit)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append("# Benchmark Human Review")
        lines.append("")
        lines.append(f"- run_id: {self._run_id}")
        lines.append(f"- timestamp_utc: {self._timestamp_utc}")
        lines.append(f"- rows: {len(review_df)}")
        lines.append("")

        if review_df.empty:
            lines.append("No benchmark rows found.")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Human review markdown written: %s", out_path)
            return

        for scope in ("local", "global"):
            subset = review_df[review_df["question_scope"] == scope]
            if subset.empty:
                continue
            lines.append(f"## Scope: {scope}")
            lines.append("")

            for question in subset["question"].drop_duplicates().tolist():
                qdf = subset[subset["question"] == question]
                lines.append(f"### Question: {question}")
                lines.append("")
                for rec in qdf.to_dict(orient="records"):
                    lines.append(f"#### {rec['pipeline']} ({rec.get('search_mode', '')})")
                    lines.append("")
                    lines.append("Answer")
                    lines.append(rec.get("answer", "") or "")
                    lines.append("")
                    lines.append("Top-k Retrieved Chunks")
                    chunks_txt = rec.get("top_k_chunks", "") or "(none)"
                    lines.append(chunks_txt)
                    lines.append("")
                    lines.append("Top-k Chunk Details (text + metadata)")
                    lines.append(rec.get("top_k_chunk_details", "") or "(none)")
                    lines.append("")
                    lines.append(
                        "Metrics: "
                        f"P@k={float(rec.get('precision_5', 0.0)):.3f}, "
                        f"R@k={float(rec.get('recall_5', 0.0)):.3f}, "
                        f"MRR={float(rec.get('mrr', 0.0)):.3f}, "
                        f"Faithfulness={float(rec.get('faithfulness', 0.0)):.3f}, "
                        f"Comprehensiveness={float(rec.get('comprehensiveness', 0.0)):.3f}, "
                        f"Diversity={float(rec.get('diversity', 0.0)):.3f}, "
                        f"Directness={float(rec.get('directness', 0.0)):.3f}, "
                        f"Empowerment={float(rec.get('empowerment', 0.0)):.3f}, "
                        f"Latency={float(rec.get('latency_s', 0.0)):.2f}s"
                    )
                    lines.append("")
                    lines.append("Judge Rationales")
                    lines.append(f"- Faithfulness: {rec.get('faithfulness_rationale', '')}")
                    lines.append(
                        f"- Comprehensiveness: {rec.get('comprehensiveness_rationale', '')}"
                    )
                    lines.append(f"- Diversity: {rec.get('diversity_rationale', '')}")
                    lines.append(f"- Directness: {rec.get('directness_rationale', '')}")
                    lines.append(f"- Empowerment: {rec.get('empowerment_rationale', '')}")
                    if rec.get("judge_validation_notes"):
                        lines.append(f"Validation Notes: {rec.get('judge_validation_notes')}")
                    lines.append("")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Human review markdown written: %s", out_path)

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
            embedding_model = getattr(self.rag.embedder, "model_name", "") or getattr(
                self.rag.embedder, "model", ""
            )
        elif self.graphrag is not None and getattr(self.graphrag, "embedder", None) is not None:
            embedding_model = getattr(self.graphrag.embedder, "model_name", "") or getattr(
                self.graphrag.embedder, "model", ""
            )

        return {
            "llm_provider": settings.llm_provider.value,
            "llm_model": llm_model,
            "embedding_provider": settings.embedding_provider.value,
            "embedding_model": embedding_model,
        }

    def _serialize_chunks(self, chunks) -> list[dict[str, object]]:
        """Convert retrieval results into JSON-serializable benchmark payloads."""
        serialized_chunks: list[dict[str, object]] = []
        for chunk in chunks or []:
            serialized_chunks.append(
                {
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "text": getattr(chunk, "text", ""),
                    "score": getattr(chunk, "score", 0.0),
                    "metadata": getattr(chunk, "metadata", None),
                }
            )
        return serialized_chunks

    # ------------------------------------------------------------------
    # Private evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_rag(self, eq: EvalQuestion, top_k: int) -> EvalRecord:
        """Evaluate standard RAG on one question."""
        t0 = time.perf_counter()
        result = self.rag.query(eq.question, top_k=top_k)
        latency = time.perf_counter() - t0 if self.compute_efficiency_tracking else 0.0

        retrieved_ids = [c.chunk_id for c in result.retrieved_chunks]
        relevant = set(eq.relevant_doc_ids)

        needs_context = (
            self.compute_efficiency_tracking
            or self.compute_faithfulness
            or self.compute_quality_judges
        )
        context = "\n\n".join(c.text for c in result.retrieved_chunks) if needs_context else ""
        meta = self._get_model_metadata()

        context_char_count = len(context) if self.compute_efficiency_tracking else 0
        estimated_context_tokens = (
            _estimate_tokens(context) if self.compute_efficiency_tracking else 0
        )

        rec = EvalRecord(
            run_id=self._run_id,
            timestamp_utc=self._timestamp_utc,
            question=eq.question,
            pipeline="rag",
            question_scope=eq.question_scope,
            search_mode="vector",
            llm_provider=meta["llm_provider"],
            llm_model=meta["llm_model"],
            embedding_provider=meta["embedding_provider"],
            embedding_model=meta["embedding_model"],
            top_k=top_k,
            answer=result.answer,
            retrieved_chunks=self._serialize_chunks(result.retrieved_chunks),
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
            relevant_doc_count=len(relevant),
            retrieved_count=len(retrieved_ids),
            relevant_hits_at_k=sum(1 for rid in retrieved_ids[:top_k] if rid in relevant),
            retrieved_chunk_ids=json.dumps(retrieved_ids, ensure_ascii=False),
            retrieved_chunk_ids_top5=json.dumps(retrieved_ids[:5], ensure_ascii=False),
            retrieved_chunk_scores=json.dumps(
                [round(float(getattr(c, "score", 0.0)), 6) for c in result.retrieved_chunks],
                ensure_ascii=False,
            ),
            retrieved_chunk_payload=json.dumps(
                _build_chunk_payload(result.retrieved_chunks),
                ensure_ascii=False,
            ),
            retrieved_chunk_payload_top5=json.dumps(
                _build_chunk_payload(result.retrieved_chunks, limit=5),
                ensure_ascii=False,
            ),
            retrieved_doc_ids=json.dumps(
                sorted({_doc_id_from_chunk_id(cid) for cid in retrieved_ids}),
                ensure_ascii=False,
            ),
            context_char_count=context_char_count,
            estimated_context_tokens=estimated_context_tokens,
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        if self.compute_faithfulness and context.strip():
            rec.faithfulness = faithfulness_score(result.answer, context)
            if self.collect_judge_explanations:
                judge = assess_faithfulness(answer=result.answer, context=context)
                rec.faithfulness = judge["score"]
                rec.faithfulness_rationale = judge["rationale"]
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
            if self.collect_judge_explanations:
                c = assess_comprehensiveness(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                d = assess_diversity(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                dr = assess_directness(
                    question=eq.question,
                    answer=result.answer,
                )
                e = assess_empowerment(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                rec.comprehensiveness = c["score"]
                rec.diversity = d["score"]
                rec.directness = dr["score"]
                rec.empowerment = e["score"]
                rec.comprehensiveness_rationale = c["rationale"]
                rec.diversity_rationale = d["rationale"]
                rec.directness_rationale = dr["rationale"]
                rec.empowerment_rationale = e["rationale"]

        if self.collect_judge_explanations:
            notes: list[str] = []
            if rec.faithfulness < 0.5 and rec.relevant_hits_at_k == 0:
                notes.append("Low faithfulness aligns with no relevant retrieval hit in top-k.")
            if rec.relevant_hits_at_k > 0 and rec.faithfulness < 0.4:
                notes.append(
                    "Potential contradiction: relevant hit exists but faithfulness is low."
                )
            rec.judge_validation_notes = " ".join(notes)

        return _validate_record(rec)

    def _evaluate_graphrag(self, eq: EvalQuestion, top_k: int, mode: str) -> EvalRecord:
        """Evaluate GraphRAG on one question with a specific search mode."""
        from src.graph_rag.retriever import SearchMode

        search_mode = SearchMode(mode)
        t0 = time.perf_counter()
        result = self.graphrag.query(eq.question, mode=search_mode, top_k=top_k)
        latency = time.perf_counter() - t0 if self.compute_efficiency_tracking else 0.0

        chunk_results = result.search_result.chunk_results if result.search_result else []
        retrieved_ids = [c.chunk_id for c in chunk_results]
        relevant = set(eq.relevant_doc_ids)

        needs_context = (
            self.compute_efficiency_tracking
            or self.compute_faithfulness
            or self.compute_quality_judges
        )
        context = ""
        if needs_context:
            context_parts = [c.text for c in chunk_results]
            if result.search_result and result.search_result.graph_context:
                context_parts.append(result.search_result.graph_context)
            for summary in result.search_result.community_summaries if result.search_result else []:
                context_parts.append(summary)
            context = "\n\n".join(context_parts)
        meta = self._get_model_metadata()

        context_char_count = len(context) if self.compute_efficiency_tracking else 0
        estimated_context_tokens = (
            _estimate_tokens(context) if self.compute_efficiency_tracking else 0
        )

        rec = EvalRecord(
            run_id=self._run_id,
            timestamp_utc=self._timestamp_utc,
            question=eq.question,
            pipeline=f"graphrag_{mode}",
            question_scope=eq.question_scope,
            search_mode=mode,
            llm_provider=meta["llm_provider"],
            llm_model=meta["llm_model"],
            embedding_provider=meta["embedding_provider"],
            embedding_model=meta["embedding_model"],
            top_k=top_k,
            answer=result.answer,
            retrieved_chunks=self._serialize_chunks(chunk_results),
            latency_s=latency,
            precision_5=precision_at_k(retrieved_ids, relevant, top_k),
            recall_5=recall_at_k(retrieved_ids, relevant, top_k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant),
            relevant_doc_count=len(relevant),
            retrieved_count=len(retrieved_ids),
            relevant_hits_at_k=sum(1 for rid in retrieved_ids[:top_k] if rid in relevant),
            retrieved_chunk_ids=json.dumps(retrieved_ids, ensure_ascii=False),
            retrieved_chunk_ids_top5=json.dumps(retrieved_ids[:5], ensure_ascii=False),
            retrieved_chunk_scores=json.dumps(
                [round(float(getattr(c, "score", 0.0)), 6) for c in chunk_results],
                ensure_ascii=False,
            ),
            retrieved_chunk_payload=json.dumps(
                _build_chunk_payload(chunk_results),
                ensure_ascii=False,
            ),
            retrieved_chunk_payload_top5=json.dumps(
                _build_chunk_payload(chunk_results, limit=5),
                ensure_ascii=False,
            ),
            retrieved_doc_ids=json.dumps(
                sorted({_doc_id_from_chunk_id(cid) for cid in retrieved_ids}),
                ensure_ascii=False,
            ),
            context_char_count=context_char_count,
            estimated_context_tokens=estimated_context_tokens,
        )

        if eq.reference_answer:
            rs = rouge_scores(result.answer, eq.reference_answer)
            rec.rouge1 = rs["rouge1"]
            rec.rougeL = rs["rougeL"]

        if self.compute_faithfulness and context.strip():
            rec.faithfulness = faithfulness_score(result.answer, context)
            if self.collect_judge_explanations:
                judge = assess_faithfulness(answer=result.answer, context=context)
                rec.faithfulness = judge["score"]
                rec.faithfulness_rationale = judge["rationale"]
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
            if self.collect_judge_explanations:
                c = assess_comprehensiveness(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                d = assess_diversity(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                dr = assess_directness(
                    question=eq.question,
                    answer=result.answer,
                )
                e = assess_empowerment(
                    question=eq.question,
                    answer=result.answer,
                    context=context,
                )
                rec.comprehensiveness = c["score"]
                rec.diversity = d["score"]
                rec.directness = dr["score"]
                rec.empowerment = e["score"]
                rec.comprehensiveness_rationale = c["rationale"]
                rec.diversity_rationale = d["rationale"]
                rec.directness_rationale = dr["rationale"]
                rec.empowerment_rationale = e["rationale"]

        if self.collect_judge_explanations:
            notes: list[str] = []
            if rec.faithfulness < 0.5 and rec.relevant_hits_at_k == 0:
                notes.append("Low faithfulness aligns with no relevant retrieval hit in top-k.")
            if rec.relevant_hits_at_k > 0 and rec.faithfulness < 0.4:
                notes.append(
                    "Potential contradiction: relevant hit exists but faithfulness is low."
                )
            rec.judge_validation_notes = " ".join(notes)

        return _validate_record(rec)
