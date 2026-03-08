# =============================================================================
# tests/test_questions.py
# Tests for evaluation question generation helpers.
# =============================================================================

from src.evaluation.questions import _extract_json_array


class TestQuestionParsing:
    def test_extract_json_array_plain(self):
        raw = '[{"question":"Q1","reference_answer":"A1","relevant_doc_ids":[]} ]'
        out = _extract_json_array(raw)
        assert len(out) == 1
        assert out[0]["question"] == "Q1"
        assert out[0]["reference_answer"] == "A1"
        assert out[0]["relevant_doc_ids"] == []

    def test_extract_json_array_markdown_wrapped(self):
        raw = """```json
        [
          {"question":"Q2","reference_answer":"A2","relevant_doc_ids":["x"]}
        ]
        ```"""
        out = _extract_json_array(raw)
        assert len(out) == 1
        assert out[0]["question"] == "Q2"
        assert out[0]["reference_answer"] == "A2"
        assert out[0]["relevant_doc_ids"] == []

    def test_extract_json_array_invalid_returns_empty(self):
        assert _extract_json_array("not json") == []
