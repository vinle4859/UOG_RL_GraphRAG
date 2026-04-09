from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import SupportsInt

import networkx as nx
import pandas as pd

_REQUIRED_ARTIFACTS = [
    "entities.parquet",
    "relationships.parquet",
    "documents.parquet",
    "text_units.parquet",
    "communities.parquet",
    "community_reports.parquet",
]

_OPTIONAL_ARTIFACTS = ["covariates.parquet"]


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, SupportsInt):
        with contextlib.suppress(Exception):
            return int(value)
    with contextlib.suppress(Exception):
        return int(str(value))
    return default


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - depends on local parquet engine
        raise RuntimeError(
            "Unable to write parquet files. Install a parquet engine, for example: pip install pyarrow"
        ) from exc


def graphml_to_graphrag_artifacts(
    graph_path: Path | str,
    output_dir: Path | str,
    *,
    use_create_final_prefix: bool = False,
) -> dict[str, Path]:
    """Convert a GraphML knowledge graph into GraphRAG visualizer parquet artifacts."""
    graph = nx.read_graphml(str(graph_path))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    node_ids = list(graph.nodes())
    node_hrid = {node_id: idx for idx, node_id in enumerate(node_ids, start=1)}

    entity_rows: list[dict] = []
    for node_id, attrs in graph.nodes(data=True):
        entity_rows.append(
            {
                "id": str(node_id),
                "human_readable_id": node_hrid[node_id],
                "title": str(attrs.get("label", node_id) or node_id),
                "type": str(attrs.get("entity_type", "UNKNOWN") or "UNKNOWN"),
                "description": str(attrs.get("description", "") or ""),
                "name": str(attrs.get("label", node_id) or node_id),
                "text_unit_ids": json.dumps([]),
                "description_embedding": json.dumps([]),
                "graph_embedding": json.dumps([]),
                "community_ids": json.dumps([]),
            }
        )

    relationship_rows: list[dict] = []
    for idx, (source, target, attrs) in enumerate(graph.edges(data=True), start=1):
        chunk_ids = attrs.get("chunk_ids", [])
        if isinstance(chunk_ids, str):
            with contextlib.suppress(Exception):
                chunk_ids = json.loads(chunk_ids)
        if not isinstance(chunk_ids, list):
            chunk_ids = []

        relationship_rows.append(
            {
                "id": f"relationship-{idx}",
                "human_readable_id": idx,
                "source": str(source),
                "target": str(target),
                "source_degree": _safe_int(graph.degree(source), 0),
                "target_degree": _safe_int(graph.degree(target), 0),
                "weight": _safe_int(attrs.get("weight", 1), 1),
                "rank": _safe_int(attrs.get("weight", 1), 1),
                "description": str(attrs.get("description", "") or ""),
                "text_unit_ids": json.dumps([str(x) for x in chunk_ids]),
                "type": str(attrs.get("relation_type", "related_to") or "related_to"),
            }
        )

    undirected = graph.to_undirected() if graph.is_directed() else graph
    partitions = [sorted(component, key=str) for component in nx.connected_components(undirected)]

    community_rows: list[dict] = []
    community_report_rows: list[dict] = []
    for idx, community_nodes in enumerate(partitions, start=1):
        community_id = f"community-{idx}"
        nodes = [str(node) for node in community_nodes]

        community_rows.append(
            {
                "id": community_id,
                "human_readable_id": idx,
                "community": idx,
                "level": 1,
                "title": f"Community {idx}",
                "summary": "Auto-generated from GraphML connected components.",
                "rank": len(nodes),
                "rank_explanation": "Derived from community size.",
                "findings": json.dumps([]),
                "full_content": "",
                "full_content_json": "",
                "period": "",
                "size": len(nodes),
                "entity_ids": json.dumps(nodes),
                "relationship_ids": json.dumps([]),
            }
        )

        community_report_rows.append(
            {
                "id": f"community-report-{idx}",
                "human_readable_id": idx,
                "community": idx,
                "parent": None,
                "level": 1,
                "title": f"Community {idx} report",
                "summary": "Auto-generated from GraphML connected components.",
                "full_content": "",
                "rank": len(nodes),
                "rank_explanation": "Derived from community size.",
                "findings": json.dumps([]),
                "full_content_json": "",
                "period": "",
                "size": len(nodes),
            }
        )

    documents_df = pd.DataFrame(
        columns=[
            "id",
            "human_readable_id",
            "title",
            "text",
            "text_unit_ids",
            "creation_date",
        ]
    )
    text_units_df = pd.DataFrame(
        columns=[
            "id",
            "human_readable_id",
            "text",
            "n_tokens",
            "document_ids",
            "entity_ids",
            "relationship_ids",
        ]
    )
    covariates_df = pd.DataFrame(
        columns=["id", "human_readable_id", "subject_id", "covariate_type"]
    )

    artifact_frames: dict[str, pd.DataFrame] = {
        "entities.parquet": pd.DataFrame(entity_rows),
        "relationships.parquet": pd.DataFrame(relationship_rows),
        "documents.parquet": documents_df,
        "text_units.parquet": text_units_df,
        "communities.parquet": pd.DataFrame(community_rows),
        "community_reports.parquet": pd.DataFrame(community_report_rows),
        "covariates.parquet": covariates_df,
    }

    written: dict[str, Path] = {}
    for base_name, frame in artifact_frames.items():
        filename = f"create_final_{base_name}" if use_create_final_prefix else base_name
        out_path = output / filename
        _write_parquet(frame, out_path)
        written[filename] = out_path

    return written
