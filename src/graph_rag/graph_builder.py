# =============================================================================
# src/graph_rag/graph_builder.py
# Construct and persist a knowledge graph from extraction results.
# =============================================================================
"""
Knowledge Graph Builder
=======================
Aggregates entity–relation triples into a unified graph.  Supports:

- **NetworkX** (in-memory, serialised to GraphML / JSON) — for prototyping.
- **Neo4j** (persistent graph DB) — for production / large graphs.

Features
--------
- Deduplicates entities by normalised name.
- Merges edge weights when the same relation appears in multiple chunks.
- Stores provenance (which chunk each edge came from).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from src.config import GraphStoreType, get_settings
from src.graph_rag.entity_extractor import Entity, ExtractionResult, Relation

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Build and query a knowledge graph of entities and relations.

    The graph is a directed multigraph where:
    - **Nodes** = entities (with attributes: type, description).
    - **Edges** = relations (with attributes: relation_type, description,
      weight, chunk_ids).
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._entity_registry: dict[str, Entity] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_extraction(self, result: ExtractionResult) -> None:
        """Merge one ExtractionResult into the graph."""
        for ent in result.entities:
            self._add_entity(ent)
        for rel in result.relations:
            self._add_relation(rel)

    def add_extractions(self, results: list[ExtractionResult]) -> None:
        """Merge multiple ExtractionResult objects."""
        for r in results:
            self.add_extraction(r)
        logger.info(
            "Graph now has %d nodes and %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _add_entity(self, entity: Entity) -> None:
        """Add or update an entity node."""
        key = entity.name.strip().lower()
        if key in self._entity_registry:
            return  # already present
        self._entity_registry[key] = entity
        self.graph.add_node(
            key,
            label=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
        )

    def _add_relation(self, relation: Relation) -> None:
        """Add or strengthen a relation edge."""
        src = relation.source.strip().lower()
        tgt = relation.target.strip().lower()

        # Ensure endpoints exist
        if src not in self.graph:
            self.graph.add_node(src, label=relation.source, entity_type="UNKNOWN")
        if tgt not in self.graph:
            self.graph.add_node(tgt, label=relation.target, entity_type="UNKNOWN")

        if self.graph.has_edge(src, tgt):
            # Increment weight and append provenance
            edge = self.graph.edges[src, tgt]
            edge["weight"] = edge.get("weight", 1) + 1
            edge.setdefault("chunk_ids", []).append(relation.chunk_id)
        else:
            self.graph.add_edge(
                src,
                tgt,
                relation_type=relation.relation_type,
                description=relation.description,
                weight=1,
                chunk_ids=[relation.chunk_id],
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_neighbours(self, entity_name: str, hops: int = 1) -> nx.DiGraph:
        """Return the ego-graph around *entity_name* up to *hops* edges away."""
        key = entity_name.strip().lower()
        if key not in self.graph:
            return nx.DiGraph()
        return nx.ego_graph(self.graph, key, radius=hops)

    def get_entity_info(self, entity_name: str) -> dict | None:
        """Return node attributes for an entity."""
        key = entity_name.strip().lower()
        if key in self.graph:
            return dict(self.graph.nodes[key])
        return None

    def top_entities(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the *n* highest-degree entities."""
        degree = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        return degree[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str = "data/graphs/knowledge_graph.graphml") -> None:
        """Serialise the graph to GraphML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # GraphML cannot store list attributes; convert chunk_ids to JSON strings
        g = self.graph.copy()
        for _, _, d in g.edges(data=True):
            if "chunk_ids" in d:
                d["chunk_ids"] = json.dumps(d["chunk_ids"])
        nx.write_graphml(g, str(path))
        logger.info("Graph saved to %s", path)

    def load(self, path: Path | str = "data/graphs/knowledge_graph.graphml") -> None:
        """Load a graph from GraphML."""
        path = Path(path)
        self.graph = nx.read_graphml(str(path))
        # Restore chunk_ids from JSON strings
        for _, _, d in self.graph.edges(data=True):
            if "chunk_ids" in d and isinstance(d["chunk_ids"], str):
                d["chunk_ids"] = json.loads(d["chunk_ids"])
        logger.info(
            "Graph loaded from %s: %d nodes, %d edges.",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
