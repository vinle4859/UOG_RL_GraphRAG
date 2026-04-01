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
import json
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _parse_judge_payload(raw: str) -> dict[str, Any]:
    """Parse either JSON judge output or legacy scalar output."""
    text = (raw or "").strip()
    if not text:
        return {"score": 0.0, "rationale": ""}

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = text[start : end + 1]
        try:
            obj = json.loads(payload)
            score = _parse_faithfulness_score(str(obj.get("score", "0")))
            rationale = str(obj.get("rationale", "")).strip()
            return {"score": score, "rationale": rationale}
        except Exception:
            pass

    return {"score": _parse_faithfulness_score(text), "rationale": ""}


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

Scoring rubric:
- 1.0: Every substantive claim is supported by context.
- 0.8: Mostly supported with minor unsupported detail.
- 0.5: Mixed support; notable unsupported or ambiguous claims.
- 0.2: Mostly unsupported, speculative, or weakly grounded.
- 0.0: Contradicts context or is unrelated.

Context:
{context}

Answer:
{answer}

Return format requirements:
- Return ONLY one decimal number between 0.0 and 1.0.
- Do not return explanation text.
- Prefer one decimal place using the rubric anchors above.
"""

COMPREHENSIVENESS_PROMPT = """\
You are an impartial judge evaluating how comprehensive an answer is given a
question and supporting context.

Scoring rubric:
- 1.0: Covers nearly all major relevant points in context.
- 0.8: Covers most major points, minor omissions.
- 0.5: Covers some points but misses important aspects.
- 0.2: Covers only a small subset of key points.
- 0.0: Very incomplete or irrelevant.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return format requirements:
- Return ONLY one decimal number between 0.0 and 1.0.
- Do not return explanation text.
- Prefer one decimal place using the rubric anchors above.
"""

DIVERSITY_PROMPT = """\
You are an impartial judge evaluating informational diversity of an answer.

Scoring rubric:
- 1.0: Covers multiple distinct relevant facets/perspectives from context.
- 0.8: Good breadth with minor redundancy.
- 0.5: Some breadth, but still narrow.
- 0.2: Mostly single-facet with little variation.
- 0.0: Repetitive, very narrow, or off-topic.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return format requirements:
- Return ONLY one decimal number between 0.0 and 1.0.
- Do not return explanation text.
- Prefer one decimal place using the rubric anchors above.
"""

DIRECTNESS_PROMPT = """\
You are an impartial judge evaluating directness of an answer.

Scoring rubric:
- 1.0: Directly answers the question with minimal detours.
- 0.8: Mostly direct with small digressions.
- 0.5: Partly direct but includes avoidable detours.
- 0.2: Mostly indirect and weakly targeted.
- 0.0: Fails to answer what was asked.

Question:
{question}

Answer:
{answer}

Return format requirements:
- Return ONLY one decimal number between 0.0 and 1.0.
- Do not return explanation text.
- Prefer one decimal place using the rubric anchors above.
"""

EMPOWERMENT_PROMPT = """\
You are an impartial judge evaluating empowerment of an answer.

Scoring rubric:
- 1.0: Enables clear understanding and actionable next steps.
- 0.8: Useful and mostly actionable.
- 0.5: Somewhat useful but limited actionability.
- 0.2: Weak utility or unclear actionability.
- 0.0: Not useful for decisions or follow-up action.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return format requirements:
- Return ONLY one decimal number between 0.0 and 1.0.
- Do not return explanation text.
- Prefer one decimal place using the rubric anchors above.
"""

FAITHFULNESS_ASSESSMENT_PROMPT = """\
You are an impartial evaluator. Score the answer for faithfulness to context.

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON with this schema:
{{"score": <0.0-1.0 number>, "rationale": "<=2 concise sentences"}}
"""

COMPREHENSIVENESS_ASSESSMENT_PROMPT = """\
You are an impartial evaluator. Score answer comprehensiveness.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON with this schema:
{{"score": <0.0-1.0 number>, "rationale": "<=2 concise sentences"}}
"""

DIVERSITY_ASSESSMENT_PROMPT = """\
You are an impartial evaluator. Score informational diversity.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON with this schema:
{{"score": <0.0-1.0 number>, "rationale": "<=2 concise sentences"}}
"""

DIRECTNESS_ASSESSMENT_PROMPT = """\
You are an impartial evaluator. Score directness.

Question:
{question}

Answer:
{answer}

Return ONLY valid JSON with this schema:
{{"score": <0.0-1.0 number>, "rationale": "<=2 concise sentences"}}
"""

EMPOWERMENT_ASSESSMENT_PROMPT = """\
You are an impartial evaluator. Score empowerment/usefulness.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON with this schema:
{{"score": <0.0-1.0 number>, "rationale": "<=2 concise sentences"}}
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
                timeout=120,
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


def _llm_judge_assessment(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Run judge prompt and parse score + rationale when available."""
    from src.config import LLMProvider, get_settings

    settings = get_settings()

    try:
        if settings.llm_provider == LLMProvider.OLLAMA:
            import requests

            ollama_model = model or settings.ollama_model
            resp = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=120,
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
                max_tokens=120,
            )
            raw = (response.choices[0].message.content or "").strip()

        parsed = _parse_judge_payload(raw)
        return {
            "score": parsed.get("score", 0.0),
            "rationale": parsed.get("rationale", ""),
            "raw": raw,
        }

    except Exception:
        logger.exception("Judge assessment failed — returning default score+rationale")
        return {"score": 0.0, "rationale": "", "raw": ""}


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


def assess_faithfulness(answer: str, context: str, model: str | None = None) -> dict[str, Any]:
    """Return faithfulness score with short rationale for human review."""
    prompt = FAITHFULNESS_ASSESSMENT_PROMPT.format(context=context, answer=answer)
    return _llm_judge_assessment(prompt, model=model)


def assess_comprehensiveness(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Return comprehensiveness score with short rationale for human review."""
    prompt = COMPREHENSIVENESS_ASSESSMENT_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    return _llm_judge_assessment(prompt, model=model)


def assess_diversity(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Return diversity score with short rationale for human review."""
    prompt = DIVERSITY_ASSESSMENT_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    return _llm_judge_assessment(prompt, model=model)


def assess_directness(
    question: str,
    answer: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Return directness score with short rationale for human review."""
    prompt = DIRECTNESS_ASSESSMENT_PROMPT.format(
        question=question,
        answer=answer,
    )
    return _llm_judge_assessment(prompt, model=model)


def assess_empowerment(
    question: str,
    answer: str,
    context: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Return empowerment score with short rationale for human review."""
    prompt = EMPOWERMENT_ASSESSMENT_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    return _llm_judge_assessment(prompt, model=model)
