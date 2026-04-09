from __future__ import annotations

import html
import logging
from pathlib import Path

import networkx as nx

from src.graph_rag.community import Community, default_community_path, load_communities

logger = logging.getLogger(__name__)

_PALETTE = [
    "#0f766e",
    "#1d4ed8",
    "#d97706",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0891b2",
    "#ea580c",
    "#334155",
    "#65a30d",
]


def build_node_community_map(communities: list[Community]) -> dict[str, int]:
    """Return a node -> community_id mapping."""
    mapping: dict[str, int] = {}
    for community in communities:
        for node in community.nodes:
            mapping[str(node)] = community.community_id
    return mapping


def sample_graph_for_visualization(
    graph: nx.Graph | nx.DiGraph,
    communities: list[Community],
    *,
    max_nodes: int = 500,
) -> tuple[nx.Graph | nx.DiGraph, list[Community]]:
    """Sample a high-signal subgraph for stable browser rendering."""
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy(), communities

    available = set(graph.nodes())
    ranked_communities = sorted(
        communities,
        key=lambda community: (len(community.nodes), bool(community.summary.strip())),
        reverse=True,
    )
    node_queues = []
    for community in ranked_communities:
        nodes = [node for node in community.nodes if node in available]
        ranked = sorted(nodes, key=lambda node: graph.degree(node), reverse=True)
        if ranked:
            node_queues.append(ranked)

    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < max_nodes and any(queue for queue in node_queues):
        for queue in node_queues:
            while queue and queue[0] in selected_set:
                queue.pop(0)
            if not queue:
                continue
            node = queue.pop(0)
            selected.append(node)
            selected_set.add(node)
            if len(selected) >= max_nodes:
                break

    if len(selected) < max_nodes:
        fallback_nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
        for node in fallback_nodes:
            if node in selected_set:
                continue
            selected.append(node)
            selected_set.add(node)
            if len(selected) >= max_nodes:
                break

    subgraph = graph.subgraph(selected).copy()
    sampled_communities: list[Community] = []
    for community in ranked_communities:
        sampled_nodes = [node for node in community.nodes if node in subgraph]
        if not sampled_nodes:
            continue
        sampled_communities.append(
            Community(
                community_id=community.community_id,
                nodes=sampled_nodes,
                summary=community.summary,
                level=community.level,
                metadata=dict(community.metadata),
            )
        )
    return subgraph, sampled_communities


def render_graph_visualization(
    graph_path: Path | str,
    *,
    community_path: Path | str | None = None,
    output_path: Path | str = "results/graph_visualizer/knowledge_graph.html",
    max_nodes: int = 500,
) -> Path:
    """Render an interactive HTML graph using a safe sampled subgraph."""
    try:
        from pyvis.network import Network
    except ModuleNotFoundError as exc:
        missing = exc.name or "pyvis"
        raise RuntimeError(
            f"Missing dependency `{missing}` required for graph visualization. "
            "Install it with `pip install pyvis ipython` before rendering graph HTML."
        ) from exc

    graph_path = Path(graph_path)
    community_path = Path(community_path) if community_path else default_community_path(graph_path)
    graph = nx.read_graphml(str(graph_path))
    communities = load_communities(community_path) if community_path.exists() else []
    node_to_community = build_node_community_map(communities)
    sampled_graph, sampled_communities = sample_graph_for_visualization(
        graph,
        communities,
        max_nodes=max_nodes,
    )
    sampled_map = build_node_community_map(sampled_communities)

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#f8fafc",
        font_color="#0f172a",
        directed=sampled_graph.is_directed(),
    )
    net.barnes_hut(gravity=-25000, spring_length=140, spring_strength=0.02, central_gravity=0.2)

    for node, attrs in sampled_graph.nodes(data=True):
        label = str(attrs.get("label", node) or node)
        description = str(attrs.get("description", "") or "")
        entity_type = str(attrs.get("entity_type", "UNKNOWN") or "UNKNOWN")
        community_id = sampled_map.get(node, node_to_community.get(node, -1))
        degree = int(sampled_graph.degree(node))
        color = _PALETTE[community_id % len(_PALETTE)] if community_id >= 0 else "#64748b"
        title = (
            f"<b>{html.escape(label)}</b><br>"
            f"Type: {html.escape(entity_type)}<br>"
            f"Degree: {degree}<br>"
            f"Community: {community_id if community_id >= 0 else 'unassigned'}"
        )
        if description:
            title += f"<br><br>{html.escape(description[:300])}"
        net.add_node(
            node,
            label=label[:42],
            title=title,
            color=color,
            size=min(14 + degree, 34),
        )

    for source, target, attrs in sampled_graph.edges(data=True):
        relation = str(attrs.get("relation_type", "related_to") or "related_to")
        weight = int(attrs.get("weight", 1) or 1)
        title = f"Relation: {html.escape(relation)}<br>" f"Weight: {weight}"
        net.add_edge(
            source,
            target,
            label=relation[:26],
            title=title,
            width=min(max(weight, 1), 6),
            color="#94a3b8",
            arrows="to" if sampled_graph.is_directed() else "",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False)
    html_text = output_path.read_text(encoding="utf-8")
    html_text = html_text.replace(
        "</body>",
        _community_sidebar(
            full_graph=graph,
            sampled_graph=sampled_graph,
            communities=sampled_communities,
        )
        + "\n</body>",
    )
    output_path.write_text(html_text, encoding="utf-8")
    logger.info(
        "Graph visualizer written to %s (%d nodes, %d edges).",
        output_path,
        sampled_graph.number_of_nodes(),
        sampled_graph.number_of_edges(),
    )
    return output_path


def _community_sidebar(
    *,
    full_graph: nx.Graph | nx.DiGraph,
    sampled_graph: nx.Graph | nx.DiGraph,
    communities: list[Community],
) -> str:
    items = []
    for community in sorted(communities, key=lambda item: len(item.nodes), reverse=True)[:12]:
        color = _PALETTE[community.community_id % len(_PALETTE)]
        summary = html.escape((community.summary or "No summary available.")[:240])
        items.append(
            "<div class='community-item'>"
            f"<div class='community-chip' style='background:{color}'></div>"
            f"<div><strong>Community {community.community_id}</strong> "
            f"({len(community.nodes)} sampled nodes)<br>{summary}</div>"
            "</div>"
        )

    items_html = "\n".join(items) if items else "<p>No community sidecar loaded.</p>"
    return f"""
<style>
body {{
  font-family: 'Segoe UI', sans-serif;
}}
#community-sidebar {{
  position: fixed;
  top: 16px;
  right: 16px;
  width: 320px;
  max-height: calc(100vh - 32px);
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid #cbd5e1;
  border-radius: 14px;
  box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
  padding: 16px;
  z-index: 999;
}}
.community-item {{
  display: flex;
  gap: 10px;
  margin-top: 12px;
  font-size: 13px;
  line-height: 1.45;
}}
.community-chip {{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  margin-top: 4px;
  flex: 0 0 auto;
}}
</style>
<aside id="community-sidebar">
  <h3 style="margin:0 0 10px 0;">Community View</h3>
  <p style="margin:0 0 10px 0;font-size:13px;">
    Full graph: {full_graph.number_of_nodes():,} nodes / {full_graph.number_of_edges():,} edges<br>
    Sampled graph: {sampled_graph.number_of_nodes():,} nodes / {sampled_graph.number_of_edges():,} edges
  </p>
  {items_html}
</aside>
"""
