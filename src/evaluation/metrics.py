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
- Comprehensiveness score (LLM-as-judge — does the answer cover key points?)
- Diversity score (LLM-as-judge — does the answer avoid narrow, repetitive framing?)
- Directness score (LLM-as-judge — does the answer address the question directly?)
- Empowerment score (LLM-as-judge — does the answer enable next-step action/understanding?)

All functions accept simple Python types so they're easy to use from
notebooks and scripts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _parse_faithfulness_score(raw: str) -> float:
    """Extract and clamp a numeric faithfulness score from raw LLM output."""
    match = re.search(r"(\d+\.?\d*)", raw)
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(1.0, score))


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


FAITHFULNESS_PROMPT = """\
You are an impartial judge evaluating whether an answer is faithfully grounded
in the provided context.  A faithful answer only makes claims that are
supported by the context — it does not hallucinate or add information.

Context:
{context}

Answer:
{answer}

Rate the faithfulness of the answer on a scale from 0.0 to 1.0:
- 1.0 = every claim is directly supported by the context
- 0.5 = some claims are supported, some are not
- 0.0 = the answer contradicts or is unrelated to the context

Respond with ONLY a single number between 0.0 and 1.0, nothing else.
"""

COMPREHENSIVENESS_PROMPT = """\
You are an impartial judge evaluating how comprehensive an answer is given a
question and supporting context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Rate comprehensiveness from 0.0 to 1.0:
- 1.0 = covers nearly all major relevant points supported by context
- 0.5 = covers some points but misses important aspects
- 0.0 = very incomplete or largely irrelevant

Respond with ONLY a single number between 0.0 and 1.0.
"""

DIVERSITY_PROMPT = """\
You are an impartial judge evaluating informational diversity of an answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Rate diversity from 0.0 to 1.0:
- 1.0 = answer spans multiple relevant facets/perspectives from context
- 0.5 = somewhat varied but narrow in scope
- 0.0 = single-facet, repetitive, or overly narrow

Respond with ONLY a single number between 0.0 and 1.0.
"""

DIRECTNESS_PROMPT = """\
You are an impartial judge evaluating directness of an answer.

Question:
{question}

Answer:
{answer}

Rate directness from 0.0 to 1.0:
- 1.0 = directly answers the question with minimal unnecessary detours
- 0.5 = partly direct but includes avoidable digressions
- 0.0 = mostly indirect or fails to answer what was asked

Respond with ONLY a single number between 0.0 and 1.0.
"""

EMPOWERMENT_PROMPT = """\
You are an impartial judge evaluating empowerment of an answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Rate empowerment from 0.0 to 1.0:
- 1.0 = leaves the user with clear understanding and actionable next steps
- 0.5 = somewhat useful but lacks clarity/actionability
- 0.0 = not useful for decision-making or follow-up action

Respond with ONLY a single number between 0.0 and 1.0.
"""


def _llm_scalar_score(prompt: str, model: str | None = None) -> float:
    """Run a scalar LLM-as-judge prompt and parse score in [0, 1]."""
    from src.config import LLMProvider, get_settings

    settings = get_settings()

    try:
        if settings.llm_provider == LLMProvider.OLLAMA:
            import requests

            ollama_model = model or settings.ollama_model
            resp = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=settings.ollama_request_timeout,
            )
            resp.raise_for_status()
            raw = resp.json()["response"].strip()
        else:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()

        score = _parse_faithfulness_score(raw)
        if score > 0.0 or raw.strip().startswith("0"):
            return score
        logger.warning("Could not parse scalar judge score from: %s", raw)
        return 0.0

    except Exception:
        logger.exception("Scalar judge scoring failed — returning 0.0")
        return 0.0


def faithfulness_score(
    answer: str,
    context: str,
    model: str | None = None,
) -> float:
    """
    LLM-as-judge: rate how faithfully the *answer* is grounded in the
    *context* on a scale of 0–1.

    Supports both OpenAI and Ollama backends based on the configured
    ``LLM_PROVIDER``.

    Parameters
    ----------
    answer : str
        The generated answer.
    context : str
        The retrieval context provided to the generator.
    model : str, optional
        LLM to use as judge.  Defaults per provider.

    Returns
    -------
    float
        Score in [0, 1].
    """
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return _llm_scalar_score(prompt, model=model)


def comprehensiveness_score(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> float:
    """LLM-as-judge score for coverage/completeness of answer."""
    prompt = COMPREHENSIVENESS_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )
    return _llm_scalar_score(prompt, model=model)


def diversity_score(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> float:
    """LLM-as-judge score for informational breadth/diversity."""
    prompt = DIVERSITY_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )
    return _llm_scalar_score(prompt, model=model)


def directness_score(
    question: str,
    answer: str,
    model: str | None = None,
) -> float:
    """LLM-as-judge score for directness of response to the question."""
    prompt = DIRECTNESS_PROMPT.format(
        question=question,
        answer=answer,
    )
    return _llm_scalar_score(prompt, model=model)


def empowerment_score(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> float:
    """LLM-as-judge score for usefulness/actionability of answer."""
    prompt = EMPOWERMENT_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    return _llm_scalar_score(prompt, model=model)
