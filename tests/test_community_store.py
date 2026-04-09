# =============================================================================
# tests/test_community_store.py
# Tests for community sidecar persistence.
# =============================================================================

from __future__ import annotations

from src.graph_rag.community import Community, load_communities, save_communities


def test_community_sidecar_round_trip(tmp_path):
    path = tmp_path / "graph.communities.json"
    communities = [
        Community(
            community_id=7,
            nodes=["ppo", "reinforcement learning"],
            summary="A reinforcement learning cluster.",
            level=0,
            metadata={"size": 2},
        )
    ]

    save_communities(communities, path, graph_path="data/graphs/knowledge_graph.graphml")
    loaded = load_communities(path)

    assert len(loaded) == 1
    assert loaded[0].community_id == 7
    assert loaded[0].nodes == ["ppo", "reinforcement learning"]
    assert loaded[0].summary == "A reinforcement learning cluster."
