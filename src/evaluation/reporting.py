# =============================================================================
# src/evaluation/reporting.py
# Human-readable benchmark summaries.
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

FALLBACK_PATTERNS = (
    "not enough context",
    "insufficient context",
    "do not have enough context",
    "doesn't contain any information",
    "does not contain any information",
    "please provide the text",
    "i need the context",
    "cannot answer",
    "can't answer",
    "unable to answer",
)


def write_markdown_report(
    records: list[dict[str, Any]],
    *,
    output_path: Path | str,
    questions_path: Path | str,
    run_id: str,
    timestamp_utc: str,
    llm_model: str | None = None,
    graphrag_modes: list[str] | None = None,
    top_k: int = 5,
) -> Path:
    """Render and persist a human-readable benchmark report."""
    report = build_markdown_report(
        records,
        questions_path=questions_path,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        llm_model=llm_model,
        graphrag_modes=graphrag_modes,
        top_k=top_k,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def build_markdown_report(
    records: list[dict[str, Any]],
    *,
    questions_path: Path | str,
    run_id: str,
    timestamp_utc: str,
    llm_model: str | None = None,
    graphrag_modes: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """Build a Markdown benchmark summary from JSON-serialisable records."""
    pipelines = sorted(
        {str(record.get("pipeline", "")) for record in records if record.get("pipeline")}
    )
    by_pipeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pipeline[str(record.get("pipeline", ""))].append(record)

    lines: list[str] = [
        f"# Benchmark Report: Run {run_id[:8]}",
        "",
        "## 1. Run Configuration",
        "",
        f"- Date generated: {datetime.now().date().isoformat()}",
        f"- Run ID: {run_id}",
        f"- Run timestamp (UTC): {timestamp_utc}",
        f"- Question file used: {questions_path}",
        f"- Top-k: {top_k}",
        f"- Pipelines evaluated: {', '.join(pipelines) if pipelines else '(none)'}",
        f"- GraphRAG modes requested: {', '.join(graphrag_modes or []) if graphrag_modes else 'default'}",
    ]
    if llm_model:
        lines.append(f"- Benchmark model override: {llm_model}")

    lines.extend(
        [
            "",
            "## 2. Tier Coverage Check",
            "",
            f"- Tier 1 (LLM judges): {_tier_one_status(records)}",
            f"- Tier 2 (efficiency tracking): {_tier_two_status(records)}",
            f"- Tier 3 (retrieval sanity metrics): {_tier_three_status(records)}",
            "",
            "## 3. Aggregate Results",
            "",
        ]
    )

    aggregate_headers = [
        "Pipeline",
        "Rows",
        "rouge1",
        "rougeL",
        "faithfulness",
        "latency_s",
        "est_context_tokens",
        "fallback_rate",
        "empty_answer_rate",
    ]
    aggregate_rows: list[list[str]] = []
    for pipeline in pipelines:
        rows = by_pipeline[pipeline]
        aggregate_rows.append(
            [
                pipeline,
                str(len(rows)),
                _fmt(_avg(rows, "rouge1")),
                _fmt(_avg(rows, "rougeL")),
                _fmt(_avg(rows, "faithfulness")),
                _fmt(_avg(rows, "latency_s")),
                _fmt(_avg(rows, "estimated_context_tokens")),
                _fmt_pct(_fallback_rate(rows)),
                _fmt_pct(_empty_answer_rate(rows)),
            ]
        )
    lines.extend(_markdown_table(aggregate_headers, aggregate_rows))

    lines.extend(
        [
            "",
            "## 4. Reliability and Failure Summary",
            "",
        ]
    )
    worst_fallback = sorted(
        ((_fallback_rate(rows), pipeline) for pipeline, rows in by_pipeline.items()),
        reverse=True,
    )
    if worst_fallback:
        lines.append(
            f"- Highest fallback rate: `{worst_fallback[0][1]}` at {_fmt_pct(worst_fallback[0][0])}."
        )
    best_latency = sorted(
        (_avg(rows, "latency_s"), pipeline) for pipeline, rows in by_pipeline.items()
    )
    if best_latency:
        lines.append(
            f"- Lowest average latency: `{best_latency[0][1]}` at {_fmt(best_latency[0][0])} s."
        )
    highest_context = sorted(
        (
            (_avg(rows, "estimated_context_tokens"), pipeline)
            for pipeline, rows in by_pipeline.items()
        ),
        reverse=True,
    )
    if highest_context:
        lines.append(
            f"- Largest average context: `{highest_context[0][1]}` at {_fmt(highest_context[0][0])} tokens."
        )
    if _tier_three_status(records).startswith("Unavailable"):
        lines.append("- Retrieval sanity metrics are still non-informative in this artifact set.")

    lines.extend(
        [
            "",
            "## 5. Best and Worst Examples",
            "",
        ]
    )
    examples = _best_and_worst_examples(records)
    if not examples:
        lines.append("- No benchmark rows were available for examples.")
    else:
        for label, record in examples:
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- Question: {record.get('question', '')}",
                    f"- Pipeline: {record.get('pipeline', '')}",
                    f"- rouge1: {_fmt(float(record.get('rouge1', 0.0) or 0.0))}",
                    f"- latency_s: {_fmt(float(record.get('latency_s', 0.0) or 0.0))}",
                    f"- Answer snippet: {_snippet(str(record.get('answer', '') or ''))}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 6. Recommendations",
            "",
        ]
    )
    lines.extend(_recommendations(by_pipeline))
    lines.append("")
    return "\n".join(lines)


def _tier_one_status(records: list[dict[str, Any]]) -> str:
    quality_fields = [
        "faithfulness",
        "comprehensiveness",
        "diversity",
        "directness",
        "empowerment",
    ]
    active = [
        field
        for field in quality_fields
        if any(float(record.get(field, 0.0) or 0.0) > 0 for record in records)
    ]
    return f"Available via {', '.join(active)}" if active else "Unavailable or all zero in this run"


def _tier_two_status(records: list[dict[str, Any]]) -> str:
    active = any(float(record.get("latency_s", 0.0) or 0.0) > 0 for record in records) or any(
        int(record.get("estimated_context_tokens", 0) or 0) > 0 for record in records
    )
    return "Available and populated" if active else "Unavailable or all zero in this run"


def _tier_three_status(records: list[dict[str, Any]]) -> str:
    active = (
        any(float(record.get("precision_5", 0.0) or 0.0) > 0 for record in records)
        or any(float(record.get("recall_5", 0.0) or 0.0) > 0 for record in records)
        or any(float(record.get("mrr", 0.0) or 0.0) > 0 for record in records)
    )
    return "Available and informative" if active else "Unavailable or all zero in this run"


def _best_and_worst_examples(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not records:
        return []
    ranked = sorted(records, key=lambda record: float(record.get("rouge1", 0.0) or 0.0))
    worst = ranked[0]
    best = ranked[-1]
    if best is worst:
        return [("Representative Example", best)]
    return [("Best Example", best), ("Worst Example", worst)]


def _recommendations(by_pipeline: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    if not by_pipeline:
        return ["- No results were available to summarise."]

    best_rouge = max(by_pipeline, key=lambda pipeline: _avg(by_pipeline[pipeline], "rouge1"))
    lines.append(
        f"- Keep `{best_rouge}` as the current lexical-quality baseline; it has the strongest average rouge1 in this run."
    )

    best_faithfulness = max(
        by_pipeline, key=lambda pipeline: _avg(by_pipeline[pipeline], "faithfulness")
    )
    if _avg(by_pipeline[best_faithfulness], "faithfulness") > 0:
        lines.append(
            f"- `{best_faithfulness}` is currently strongest on faithfulness and is worth re-checking against latency and fallback rates."
        )

    highest_fallback = max(by_pipeline, key=lambda pipeline: _fallback_rate(by_pipeline[pipeline]))
    if _fallback_rate(by_pipeline[highest_fallback]) > 0.2:
        lines.append(
            f"- Investigate `{highest_fallback}` before using it in overnight runs; fallback-style answers appear in {_fmt_pct(_fallback_rate(by_pipeline[highest_fallback]))} of rows."
        )

    if all(_avg(rows, "precision_5") == 0.0 for rows in by_pipeline.values()):
        lines.append(
            "- Populate or validate `relevant_doc_ids` so Tier 3 retrieval metrics become meaningful."
        )

    if not lines:
        lines.append("- No specific recommendation heuristics were triggered for this run.")
    return lines


def _avg(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values)


def _fallback_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    matches = sum(1 for row in rows if _is_fallback_answer(str(row.get("answer", "") or "")))
    return matches / len(rows)


def _empty_answer_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    empty = sum(1 for row in rows if not str(row.get("answer", "") or "").strip())
    return empty / len(rows)


def _is_fallback_answer(answer: str) -> bool:
    lowered = answer.strip().lower()
    return any(pattern in lowered for pattern in FALLBACK_PATTERNS)


def _fmt(value: float) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _snippet(answer: str, limit: int = 240) -> str:
    compact = " ".join(answer.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["- No rows available."]
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    output = [
        "|" + "|".join(headers) + "|",
        separator,
    ]
    for row in rows:
        output.append("|" + "|".join(row) + "|")
    return output
