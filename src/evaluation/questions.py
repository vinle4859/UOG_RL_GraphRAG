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
import re
from pathlib import Path

from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


QUESTION_GEN_SYSTEM_PROMPT = """\
You generate evaluation questions for benchmarking RAG systems.
Given one research abstract, produce diverse questions that test:
- factual lookup
- method understanding
- comparative reasoning
- synthesis/sensemaking

Return ONLY valid JSON as an array of objects:
[
  {"question": "...", "reference_answer": "...", "relevant_doc_ids": []}
]

Rules:
- Use concise but complete reference answers.
- Keep relevant_doc_ids as an empty list.
- No markdown, no extra text.
"""


def _extract_json_array(raw: str) -> list[dict]:
    """Extract and parse a JSON array from potentially noisy LLM output."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    payload = cleaned[start : end + 1]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    normalised: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("reference_answer", "")).strip()
        if not q:
            continue
        normalised.append(
            {
                "question": q,
                "reference_answer": a,
                "relevant_doc_ids": [],
            }
        )
    return normalised


def _call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str, model: str) -> str:
    import requests

    settings = get_settings()
    final_prompt = f"{QUESTION_GEN_SYSTEM_PROMPT}\n\n{prompt}"
    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": model, "prompt": final_prompt, "stream": False},
        timeout=settings.ollama_request_timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


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

    Returns a list of dicts with schema:
    {"question": str, "reference_answer": str, "relevant_doc_ids": []}

    Notes
    -----
    - Automatically routes to OpenAI or Ollama based on ``LLM_PROVIDER``.
    - Deduplicates by question text (case-insensitive).
    """
    settings = get_settings()
    model_name = model
    if settings.llm_provider == LLMProvider.OLLAMA and model == "gpt-4o-mini":
        model_name = settings.ollama_model

    all_items: list[dict] = []
    seen_questions: set[str] = set()

    for abstract in abstracts:
        text = (abstract or "").strip()
        if not text:
            continue

        user_prompt = (
            f"Generate {n_per_abstract} evaluation questions from this abstract.\n\n"
            f"Abstract:\n{text}"
        )

        try:
            if settings.llm_provider == LLMProvider.OLLAMA:
                raw = _call_ollama(user_prompt, model_name)
            else:
                raw = _call_openai(user_prompt, model_name)
        except Exception:
            logger.exception("Question generation failed for one abstract")
            continue

        parsed = _extract_json_array(raw)
        for item in parsed:
            key = item["question"].strip().lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            all_items.append(item)

    logger.info("Generated %d unique evaluation questions.", len(all_items))
    return all_items
