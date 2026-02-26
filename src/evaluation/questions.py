# =============================================================================
# src/evaluation/questions.py
# Helpers for creating and managing evaluation question sets.
# =============================================================================
"""
Evaluation Question Management
===============================
Utilities for generating, loading, and saving evaluation question sets.

Two strategies to create evaluation questions:
1. **Manual** — Domain experts write questions with reference answers.
2. **LLM-generated** — Feed paper abstracts to an LLM and ask it to
   generate QA pairs.  Filter and validate manually.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_template_questions(output_path: Path | str = "data/eval_questions.json") -> None:
    """
    Write a starter template with example evaluation questions.

    Edit and expand this file for your actual evaluation.
    """
    template = [
        {
            "question": "What is Proximal Policy Optimization (PPO)?",
            "reference_answer": (
                "PPO is a policy gradient method for reinforcement learning that "
                "uses a clipped surrogate objective to ensure stable updates."
            ),
            "relevant_doc_ids": [],
        },
        {
            "question": "How does attention mechanism work in Transformers?",
            "reference_answer": (
                "The attention mechanism computes a weighted sum of value vectors, "
                "where weights are derived from the compatibility of query and key "
                "vectors using scaled dot-product attention."
            ),
            "relevant_doc_ids": [],
        },
        {
            "question": "What are the main challenges of graph neural networks?",
            "reference_answer": "",
            "relevant_doc_ids": [],
        },
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    logger.info("Template questions written to %s", output_path)


def generate_questions_from_abstracts(
    abstracts: list[str],
    n_per_abstract: int = 2,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Use an LLM to generate QA pairs from paper abstracts.

    .. warning::
       Not yet implemented — returns an empty list.

    TODO
    ----
    - [ ] Implement the LLM prompt for QA generation.
    - [ ] Add deduplication / quality filtering.
    """
    logger.warning("generate_questions_from_abstracts is a stub.")
    return []
