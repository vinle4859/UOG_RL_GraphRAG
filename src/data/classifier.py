# =============================================================================
# src/data/classifier.py
# Optional title/abstract clustering for sub-field classification.
# =============================================================================
"""
Paper Sub-field Classifier
===========================
Cluster or classify papers by their research sub-field.  Two strategies are
provided:

1. **Embedding-based clustering** — Embed titles (or abstracts) with
   Sentence-Transformers, then use KMeans / HDBSCAN to discover clusters.
2. **LLM-based classification** — Send titles to an LLM with a taxonomy
   prompt and let it assign categories.

This is useful for stratified evaluation: checking whether RAG / GraphRAG
performance varies across sub-fields (e.g. RL, NLP, CV).

.. note::
   This module is *optional* and does not block the main RAG / GraphRAG
   pipelines.

TODO
----
- [ ] Decide on taxonomy (manual list vs. discovered clusters).
- [ ] Evaluate clustering quality (silhouette score, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedPaper:
    """A paper with an assigned sub-field label."""
    doc_id: str
    title: str
    label: str
    confidence: float = 0.0


def cluster_by_embeddings(
    titles: list[str],
    n_clusters: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[int]:
    """
    Embed *titles* and cluster with KMeans.

    Parameters
    ----------
    titles : list[str]
        Paper titles (or short abstracts).
    n_clusters : int
        Number of clusters to form.
    model_name : str
        Sentence-transformer model for embedding.

    Returns
    -------
    list[int]
        Cluster label for each title.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    logger.info("Embedding %d titles with %s …", len(titles), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles, show_progress_bar=True)

    logger.info("Running KMeans with k=%d …", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(np.array(embeddings))
    return labels.tolist()


# ---------------------------------------------------------------------------
# Placeholder for LLM-based classification
# ---------------------------------------------------------------------------

def classify_with_llm(titles: list[str], taxonomy: list[str] | None = None) -> list[str]:
    """
    Use an LLM to assign each title to a sub-field from *taxonomy*.

    .. warning::
       Not yet implemented — returns "unknown" for every title.

    Parameters
    ----------
    titles : list[str]
        Paper titles.
    taxonomy : list[str], optional
        Allowed category names.  If None, the LLM will invent labels.

    Returns
    -------
    list[str]
        Category label per title.
    """
    # TODO: Implement via LangChain / OpenAI structured output.
    logger.warning("classify_with_llm is a stub — returning 'unknown' for all titles.")
    return ["unknown"] * len(titles)
