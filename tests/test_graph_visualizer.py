# =============================================================================
# tests/test_graph_visualizer.py
# Tests for graph visualization sampling helpers.
# =============================================================================

from __future__ import annotations

import networkx as nx

from src.graph_rag.community import Community
from src.visualization.graph_visualizer import (
    build_node_community_map,
    sample_graph_for_visualization,
)


def test_build_node_community_map():
    communities = [
        Community(community_id=1, nodes=["ppo", "rl"]),
        Community(community_id=2, nodes=["transformer"]),
    ]
    mapping = build_node_community_map(communities)

    assert mapping["ppo"] == 1
    assert mapping["transformer"] == 2


def test_sample_graph_for_visualization_respects_limit():
    graph = nx.DiGraph()
    for idx in range(6):
        graph.add_node(f"n{idx}", label=f"Node {idx}")
    graph.add_edges_from(
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n0"),
            ("n3", "n4"),
            ("n4", "n5"),
        ]
    )
    communities = [
        Community(community_id=1, nodes=["n0", "n1", "n2"], summary="A"),
        Community(community_id=2, nodes=["n3", "n4", "n5"], summary="B"),
    ]

    sampled_graph, sampled_communities = sample_graph_for_visualization(
        graph,
        communities,
        max_nodes=4,
    )

    assert sampled_graph.number_of_nodes() <= 4
    sampled_nodes = set(sampled_graph.nodes())
    assert any(node in sampled_nodes for node in {"n0", "n1", "n2"})
    assert any(node in sampled_nodes for node in {"n3", "n4", "n5"})
    assert len(sampled_communities) >= 2
