# =============================================================================
# src/cli.py
# Command-line interface for common pipeline operations.
# =============================================================================
"""
CLI — ``rag-bench``
===================
After installation (``pip install -e .``), run::

    rag-bench --help
    rag-bench index            # Index documents for standard RAG
    rag-bench build-graph      # Build the full GraphRAG pipeline
    rag-bench query "What is PPO?"
    rag-bench benchmark data/eval_questions.json

Available commands
------------------
- ``index``       — chunk + embed all documents in ``data/processed/``.
- ``build-graph`` — run entity extraction, graph build, and community detection.
- ``query``       — ask a single question via RAG or GraphRAG.
- ``benchmark``   — evaluate both pipelines on a question set and save a CSV.
"""

from __future__ import annotations

import click

from src.utils.logging_setup import setup_logging


@click.group()
@click.option("--log-level", default="INFO", help="Logging level.")
def main(log_level: str):
    """RAG vs GraphRAG benchmarking toolkit."""
    setup_logging(log_level)


@main.command()
@click.option(
    "--data-dir",
    default=None,
    help="Directory with .json or .txt files (default: data/processed/).",
)
def index(data_dir: str | None):
    """Index all documents for standard RAG."""
    from src.rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    n = pipe.index(data_dir=data_dir)
    click.echo(f"Indexed {n} chunks.")


@main.command()
@click.argument("question")
@click.option(
    "--data-dir",
    default=None,
    help="Directory with .json or .txt files (default: data/processed/).",
)
@click.option("--top-k", default=5, help="Number of chunks to retrieve.")
def retrieve(question: str, data_dir: str | None, top_k: int):
    """Retrieve relevant chunks WITHOUT calling an LLM (no API key needed)."""
    from src.data.chunker import TokenChunker
    from src.data.loader import load_documents
    from src.rag.embedder import get_embedder
    from src.rag.retriever import Retriever
    from src.rag.vectorstore import FAISSVectorStore

    # Load & index into in-memory FAISS
    docs = load_documents(data_dir)
    if not docs:
        click.echo("No documents found. Check --data-dir or data/processed/.")
        return
    click.echo(f"Loaded {len(docs)} documents.")

    chunker = TokenChunker()
    chunks = chunker.chunk_documents(docs)
    embedder = get_embedder()
    store = FAISSVectorStore()

    embeddings = embedder.embed([c.text for c in chunks])
    store.add(
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        embeddings,
        [{"doc_id": c.doc_id, "title": c.metadata.get("title", "")} for c in chunks],
    )
    click.echo(f"Indexed {store.count()} chunks.")

    # Retrieve
    retriever = Retriever(embedder=embedder, vectorstore=store, top_k=top_k)
    results = retriever.retrieve(question)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Question: {question}")
    click.echo(f"Top {len(results)} retrieved chunks:")
    click.echo(f"{'=' * 60}")
    for i, r in enumerate(results, 1):
        doc_id = r.chunk_id.split("__")[0]
        title = (r.metadata or {}).get("title", "")
        click.echo(f"\n--- #{i}  score={r.score:.4f}  doc={doc_id} ---")
        if title:
            click.echo(f"Title: {title}")
        click.echo(r.text[:500])


@main.command("build-graph")
@click.option("--no-summarise", is_flag=True, help="Skip community summarisation.")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Process only the first N documents (useful for testing).",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Skip already-indexed docs and resume extraction from checkpoint.",
)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(),
    help="Explicit path for the extraction checkpoint JSONL file.",
)
def build_graph(no_summarise: bool, limit: int | None, resume: bool, checkpoint: str | None):
    """Build the full GraphRAG pipeline (extract → graph → communities)."""
    from src.graph_rag.pipeline import GraphRAGPipeline

    pipe = GraphRAGPipeline()
    pipe.build(
        summarise=not no_summarise,
        limit=limit,
        resume=resume,
        checkpoint_path=checkpoint,
    )
    click.echo("GraphRAG build complete.")


@main.command()
@click.argument("question")
@click.option("--pipeline", type=click.Choice(["rag", "graphrag"]), default="rag")
@click.option("--top-k", default=5, help="Number of chunks to retrieve.")
def query(question: str, pipeline: str, top_k: int):
    """Ask a question using the specified pipeline."""
    if pipeline == "rag":
        from src.rag.pipeline import RAGPipeline

        result = RAGPipeline().query(question, top_k=top_k)
    else:
        from src.graph_rag.pipeline import GraphRAGPipeline

        pipe = GraphRAGPipeline()
        pipe.load_graph()
        result = pipe.query(question, top_k=top_k)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Pipeline: {pipeline}")
    click.echo(f"Question: {question}")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nAnswer:\n{result.answer}")
    # Handle both RAGResult (retrieved_chunks) and GraphRAGResult (search_result)
    if hasattr(result, "retrieved_chunks"):
        chunks = result.retrieved_chunks
    elif hasattr(result, "search_result") and result.search_result:
        chunks = result.search_result.chunk_results
    else:
        chunks = []
    click.echo(f"\n--- Retrieved {len(chunks)} chunks ---")
    for i, r in enumerate(chunks, 1):
        doc_id = r.chunk_id.split("__")[0]
        click.echo(f"  #{i} score={r.score:.4f} doc={doc_id} chunk={r.chunk_id}")


@main.command()
@click.argument("questions_path")
@click.option("--output", default="results/benchmark_report.csv", help="Output CSV path.")
@click.option(
    "--graphrag-mode",
    "graphrag_modes",
    multiple=True,
    type=click.Choice(["local", "global", "graph_only", "hybrid"]),
    help="GraphRAG mode(s) to evaluate. Repeat option for multiple modes.",
)
@click.option(
    "--faithfulness/--no-faithfulness",
    default=False,
    help="Enable LLM-as-judge faithfulness scoring (slower, extra LLM calls).",
)
@click.option(
    "--quality-judges/--no-quality-judges",
    default=False,
    help="Enable comprehensiveness and diversity LLM judges (slowest mode).",
)
@click.option(
    "--jsonl-log/--no-jsonl-log",
    default=True,
    help="Write per-run JSONL benchmark log under results/benchmark_runs/.",
)
def benchmark(
    questions_path: str,
    output: str,
    graphrag_modes: tuple[str, ...],
    faithfulness: bool,
    quality_judges: bool,
    jsonl_log: bool,
):
    """Run side-by-side benchmark evaluation."""
    from pathlib import Path

    from src.evaluation.benchmark import BenchmarkRunner
    from src.graph_rag.pipeline import GraphRAGPipeline
    from src.rag.pipeline import RAGPipeline

    rag_pipe = RAGPipeline()

    graphrag_pipe = GraphRAGPipeline()
    graphrag_pipe.load_graph()

    modes = list(graphrag_modes) if graphrag_modes else None
    runner = BenchmarkRunner(
        rag_pipeline=rag_pipe,
        graphrag_pipeline=graphrag_pipe,
        graphrag_modes=modes,
        compute_faithfulness=faithfulness,
        compute_quality_judges=quality_judges,
    )
    df = runner.run(questions_path, write_jsonl_log=jsonl_log)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    click.echo(f"Report saved to {output}")
    click.echo(df.to_string())


if __name__ == "__main__":
    main()
