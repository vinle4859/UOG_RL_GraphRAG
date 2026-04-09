# =============================================================================
# src/graph_rag/community.py
# Community detection and community-level summarisation.
# =============================================================================
"""
Community Detection & Summarisation
====================================
Partitions the knowledge graph into communities (clusters of densely
connected entities) and generates a natural-language summary for each
community.  These summaries are used at query time for "global search"
over the entire corpus.

Algorithms
----------
- **Leiden** (preferred) — via ``leidenalg`` + ``igraph``.  Produces
  high-quality, hierarchical communities.
- **Louvain** (fallback) — available in ``networkx.community``.

Design Decision
---------------
We chose **Leiden** as the default because Microsoft's GraphRAG paper
demonstrates its superiority for this use-case.  If ``leidenalg`` is
not installed, we fall back to Louvain gracefully.

TODO
----
- [ ] Implement hierarchical community levels (coarse → fine).
- [ ] Store community summaries in the vector store for retrieval.
- [ ] Add community-level metrics (modularity, coverage).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class Community:
    """
    A detected community in the knowledge graph.

    Attributes
    ----------
    community_id : int
        Unique identifier.
    nodes : list[str]
        Entity keys belonging to this community.
    summary : str
        LLM-generated natural-language summary of the community.
    level : int
        Hierarchy level (0 = finest).
    metadata : dict
        Extra info (modularity contribution, size, etc.).
    """

    community_id: int
    nodes: list[str] = field(default_factory=list)
    summary: str = ""
    level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def default_community_path(graph_path: Path | str) -> Path:
    """Return the default JSON sidecar path for a GraphML file."""
    return Path(graph_path).with_suffix(".communities.json")


def community_to_dict(community: Community) -> dict[str, Any]:
    """Serialize a Community to a JSON-compatible dict."""
    metadata = dict(community.metadata)
    metadata.setdefault("size", len(community.nodes))
    return {
        "community_id": community.community_id,
        "nodes": sorted(set(community.nodes)),
        "summary": community.summary,
        "level": community.level,
        "size": len(community.nodes),
        "metadata": metadata,
    }


def community_from_dict(payload: dict[str, Any]) -> Community:
    """Deserialize a Community from a JSON payload."""
    nodes = [str(node) for node in payload.get("nodes", [])]
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("size", len(nodes))
    return Community(
        community_id=int(payload.get("community_id", 0)),
        nodes=nodes,
        summary=str(payload.get("summary", "") or ""),
        level=int(payload.get("level", 0)),
        metadata=metadata,
    )


def save_communities(
    communities: list[Community],
    path: Path | str,
    *,
    graph_path: Path | str | None = None,
) -> Path:
    """Persist detected communities to a JSON sidecar."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_path": str(graph_path) if graph_path is not None else "",
        "community_count": len(communities),
        "communities": [community_to_dict(community) for community in communities],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Community sidecar saved to %s (%d communities).", output_path, len(communities))
    return output_path


def load_communities(path: Path | str) -> list[Community]:
    """Load community metadata from a JSON sidecar."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    communities = [
        community_from_dict(item)
        for item in payload.get("communities", [])
        if isinstance(item, dict)
    ]
    logger.info("Loaded %d communities from %s.", len(communities), input_path)
    return communities


def has_community_summaries(communities: list[Community]) -> bool:
    """Return True when at least one community summary is populated."""
    return any(community.summary.strip() for community in communities)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_communities_leiden(
    graph: nx.Graph | nx.DiGraph,
    resolution: float = 1.0,
) -> list[Community]:
    """
    Run the Leiden algorithm on *graph*.

    Requires ``leidenalg`` and ``igraph`` packages::

        pip install leidenalg python-igraph

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        The knowledge graph.
    resolution : float
        Higher values → more / smaller communities.

    Returns
    -------
    list[Community]
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        logger.warning(
            "leidenalg/igraph not installed — falling back to Louvain. "
            "Install with: pip install leidenalg python-igraph"
        )
        return detect_communities_louvain(graph)

    # Convert NetworkX → igraph
    undirected = graph.to_undirected() if graph.is_directed() else graph
    mapping = {node: i for i, node in enumerate(undirected.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}

    ig_graph = ig.Graph(
        n=len(mapping),
        edges=[(mapping[u], mapping[v]) for u, v in undirected.edges()],
        directed=False,
    )

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )

    communities: list[Community] = []
    for cid, members in enumerate(partition):
        nodes = [reverse_mapping[m] for m in members]
        communities.append(Community(community_id=cid, nodes=nodes))

    logger.info("Leiden detected %d communities.", len(communities))
    return communities


def detect_communities_louvain(graph: nx.Graph | nx.DiGraph) -> list[Community]:
    """
    Fallback community detection using NetworkX's built-in Louvain.

    Available in networkx >= 3.0.
    """
    undirected = graph.to_undirected() if graph.is_directed() else graph
    from networkx.algorithms.community import louvain_communities

    partition = louvain_communities(undirected, seed=42)
    communities = [
        Community(community_id=cid, nodes=list(members)) for cid, members in enumerate(partition)
    ]
    logger.info("Louvain detected %d communities.", len(communities))
    return communities


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarise_community(
    community: Community,
    graph: nx.DiGraph,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a natural-language summary of a community by prompting an LLM
    with the entity names, types, and relations within the community.

    Parameters
    ----------
    community : Community
        The community to summarise.
    graph : nx.DiGraph
        The full knowledge graph (used to look up node/edge attributes).
    model : str
        LLM model name.

    Returns
    -------
    str
        A concise summary paragraph.
    """
    # Build a textual description of the community's subgraph
    lines = []
    for node in community.nodes:
        attrs = graph.nodes.get(node, {})
        label = attrs.get("label", node)
        etype = attrs.get("entity_type", "")
        lines.append(f"Entity: {label} (type: {etype})")

    subgraph = graph.subgraph(community.nodes)
    for u, v, data in subgraph.edges(data=True):
        u_label = graph.nodes[u].get("label", u)
        v_label = graph.nodes[v].get("label", v)
        rel = data.get("relation_type", "related_to")
        lines.append(f"Relation: {u_label} --[{rel}]--> {v_label}")

    context = "\n".join(lines)

    prompt = (
        "The following entities and relations belong to the same research "
        "community cluster. Write a concise summary (3–5 sentences) "
        "describing what this community is about, the key topics, and how "
        "the entities relate:\n\n"
        f"{context}\n\nSummary:"
    )

    from src.config import LLMProvider, get_settings

    settings = get_settings()

    if settings.llm_provider == LLMProvider.OLLAMA:
        from src.utils.ollama_client import ollama_post

        ollama_model = model if model != "gpt-4o-mini" else settings.ollama_model
        data = ollama_post(
            f"{settings.ollama_base_url}/api/generate",
            payload={"model": ollama_model, "prompt": prompt, "stream": False},
            read_timeout=settings.ollama_request_timeout,
        )
        summary = data["response"].strip()
    else:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()

    community.summary = summary
    return summary
