# =============================================================================
# tests/test_benchmark.py
# Tests for benchmark runner utilities and output shape.
# =============================================================================

from __future__ import annotations

import json

from src.evaluation.benchmark import BenchmarkRunner, _estimate_tokens


class _DummyChunk:
    def __init__(self, chunk_id: str, text: str):
        self.chunk_id = chunk_id
        self.text = text


class _DummyRAGResult:
    def __init__(self):
        self.answer = "Test answer"
        self.retrieved_chunks = [_DummyChunk("d1__c0", "Some context text")]


class _DummyRAGPipeline:
    def query(self, question: str, top_k: int = 5):  # noqa: ARG002
        return _DummyRAGResult()


def test_estimate_tokens_non_empty():
    assert _estimate_tokens("hello world") >= 1


def test_benchmark_writes_jsonl(tmp_path):
    qpath = tmp_path / "questions.json"
    qpath.write_text(
        json.dumps(
            [
                {
                    "question": "What is PPO?",
                    "reference_answer": "PPO is an RL algorithm.",
                    "relevant_doc_ids": ["d1__c0"],
                }
            ]
        ),
        encoding="utf-8",
    )

    runner = BenchmarkRunner(rag_pipeline=_DummyRAGPipeline())
    log_path = tmp_path / "bench.jsonl"

    df = runner.run(qpath, top_k=1, write_jsonl_log=True, jsonl_log_path=log_path)

    assert len(df) == 1
    assert "run_id" in df.columns
    assert "context_char_count" in df.columns
    assert "retrieved_chunks" in df.columns
    assert "comprehensiveness" in df.columns
    assert "diversity" in df.columns
    assert "directness" in df.columns
    assert "empowerment" in df.columns

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["pipeline"] == "rag"
    assert payload["retrieved_chunks"][0]["chunk_id"] == "d1__c0"
    assert payload["retrieved_chunks"][0]["text"] == "Some context text"
