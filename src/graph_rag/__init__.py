# =============================================================================
# src/graph_rag/__init__.py
# Graph RAG pipeline sub-package.
# =============================================================================
"""
Graph RAG Pipeline
==================
Implements a self-built GraphRAG system inspired by Microsoft's GraphRAG paper.

The flow differs from standard RAG:

1. **Entity & Relation Extraction** — Use an LLM to extract entities and
   relationships from each document chunk.
2. **Knowledge Graph Construction** — Build a graph (NetworkX or Neo4j)
   from the extracted triples.
3. **Community Detection** — Partition the graph into communities (Leiden /
   Louvain) and generate community summaries.
4. **Graph-Augmented Retrieval** — At query time, combine vector similarity
   with graph-structural signals (community membership, centrality,
   multi-hop neighbours) for richer context.
5. **Generation** — Answer using the enriched context.

Modules
-------
- ``entity_extractor`` — LLM-based NER & relation extraction.
- ``graph_builder``    — Construct and persist the knowledge graph.
- ``community``        — Community detection and summarisation.
- ``retriever``        — Graph-aware retrieval.
- ``pipeline``         — End-to-end orchestration.
"""
