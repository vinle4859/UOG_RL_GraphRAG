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
def index():
    """Index all documents for standard RAG."""
    from src.rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    n = pipe.index()
    click.echo(f"Indexed {n} chunks.")


@main.command("build-graph")
@click.option("--no-summarise", is_flag=True, help="Skip community summarisation.")
def build_graph(no_summarise: bool):
    """Build the full GraphRAG pipeline (extract → graph → communities)."""
    from src.graph_rag.pipeline import GraphRAGPipeline

    pipe = GraphRAGPipeline()
    pipe.build(summarise=not no_summarise)
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
    click.echo(result.answer)


@main.command()
@click.argument("questions_path")
@click.option("--output", default="results/benchmark_report.csv", help="Output CSV path.")
def benchmark(questions_path: str, output: str):
    """Run side-by-side benchmark evaluation."""
    from pathlib import Path
    from src.evaluation.benchmark import BenchmarkRunner
    from src.rag.pipeline import RAGPipeline
    from src.graph_rag.pipeline import GraphRAGPipeline

    rag_pipe = RAGPipeline()

    graphrag_pipe = GraphRAGPipeline()
    graphrag_pipe.load_graph()

    runner = BenchmarkRunner(rag_pipeline=rag_pipe, graphrag_pipeline=graphrag_pipe)
    df = runner.run(questions_path)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    click.echo(f"Report saved to {output}")
    click.echo(df.to_string())


if __name__ == "__main__":
    main()
