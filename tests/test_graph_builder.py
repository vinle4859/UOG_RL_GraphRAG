# =============================================================================
# tests/test_graph_builder.py
# Tests for the knowledge graph builder.
# =============================================================================
"""
Tests — Knowledge Graph Builder
================================
"""

from src.graph_rag.entity_extractor import Entity, ExtractionResult, Relation
from src.graph_rag.graph_builder import KnowledgeGraph


class TestKnowledgeGraph:
    """Test graph construction and queries."""

    def _make_extraction(self) -> ExtractionResult:
        return ExtractionResult(
            chunk_id="test_chunk_0",
            entities=[
                Entity(name="PPO", entity_type="Algorithm"),
                Entity(name="Reinforcement Learning", entity_type="Concept"),
            ],
            relations=[
                Relation(
                    source="PPO",
                    target="Reinforcement Learning",
                    relation_type="is_part_of",
                    chunk_id="test_chunk_0",
                ),
            ],
        )

    def test_add_extraction(self):
        kg = KnowledgeGraph()
        kg.add_extraction(self._make_extraction())
        assert kg.graph.number_of_nodes() == 2
        assert kg.graph.number_of_edges() == 1

    def test_deduplication(self):
        kg = KnowledgeGraph()
        ext = self._make_extraction()
        kg.add_extraction(ext)
        kg.add_extraction(ext)  # same entities again
        assert kg.graph.number_of_nodes() == 2
        # Edge weight should be 2
        edge = kg.graph.edges["ppo", "reinforcement learning"]
        assert edge["weight"] == 2

    def test_neighbours(self):
        kg = KnowledgeGraph()
        kg.add_extraction(self._make_extraction())
        ego = kg.get_neighbours("PPO", hops=1)
        assert "reinforcement learning" in ego.nodes()

    def test_top_entities(self):
        kg = KnowledgeGraph()
        kg.add_extraction(self._make_extraction())
        top = kg.top_entities(n=5)
        assert len(top) == 2
