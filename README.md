# UOG RL GraphRAG — Benchmarking GraphRAG vs Standard RAG

Benchmarking a **self-built GraphRAG** system against **standard RAG** on ~1,000 ArXiv research papers.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Platform & Tool Decisions](#platform--tool-decisions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Team Workflow (Git)](#team-workflow-git)
- [Evaluation](#evaluation)
- [Roadmap / TODOs](#roadmap--todos)

---

## Overview

| Aspect | Standard RAG | Graph RAG |
|--------|-------------|-----------|
| **Knowledge representation** | Flat vector index | Knowledge graph + vector index |
| **Retrieval** | Top-k cosine similarity | Graph traversal + vector similarity |
| **Strengths** | Simple, fast, well-understood | Multi-hop reasoning, entity relationships |
| **Weaknesses** | No structural awareness | Higher build cost, LLM extraction overhead |

This project builds **both** pipelines from scratch (no Microsoft GraphRAG library) and evaluates them head-to-head.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Papers │────▶│  Chunking    │────▶│  Embedding   │
│  (~1000 txt)│     │  (tiktoken)  │     │  (ST / OAI)  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌───────────────────────────┼───────────────────┐
                    │                           │                   │
              ┌─────▼─────┐            ┌────────▼───────┐          │
              │ Vector DB │            │ Entity Extract │          │
              │ (Chroma)  │            │ (LLM → triples)│          │
              └─────┬─────┘            └────────┬───────┘          │
                    │                           │                   │
              ┌─────▼─────┐            ┌────────▼───────┐          │
              │ Standard  │            │ Knowledge Graph│          │
              │ RAG       │            │ (NetworkX)     │          │
              │ Retriever │            └────────┬───────┘          │
              └─────┬─────┘                     │                   │
                    │                  ┌────────▼───────┐          │
                    │                  │ Community Det. │          │
                    │                  │ (Leiden/Louvain)│          │
                    │                  └────────┬───────┘          │
                    │                           │                   │
                    │                  ┌────────▼───────┐          │
                    │                  │ Graph RAG      │          │
                    │                  │ Retriever      │◀─────────┘
                    │                  └────────┬───────┘
                    │                           │
              ┌─────▼───────────────────────────▼──────┐
              │         Evaluation & Benchmark         │
              │ (LLM Judge + Efficiency + Golden Set) │
              └────────────────────────────────────────┘
```

---

## Project Structure

```
UOG_RL_GraphRAG/
├── .env.example                # Environment variable template
├── .gitignore
├── .pre-commit-config.yaml     # Code quality hooks
├── pyproject.toml              # Dependencies, build config, tool settings
├── README.md                   # ← You are here
├── CONTRIBUTING.md             # Team collaboration guide
│
├── src/                        # All source code
│   ├── __init__.py
│   ├── config.py               # Centralised settings (pydantic-settings)
│   ├── cli.py                  # CLI entry point (click)
│   │
│   ├── data/                   # Data ingestion & loading
│   │   ├── loader.py           # Load .json / .txt files into Documents
│   │   ├── preprocessor.py     # REFERENCE ONLY — JSON schema for PDF extraction team
│   │   ├── chunker.py          # Token-aware text chunking
│   │   └── classifier.py       # Optional sub-field clustering
│   │
│   ├── rag/                    # Standard RAG pipeline
│   │   ├── embedder.py         # Embedding model abstraction
│   │   ├── vectorstore.py      # Vector DB abstraction (Chroma/FAISS)
│   │   ├── retriever.py        # Top-k retrieval
│   │   ├── generator.py        # LLM answer generation
│   │   └── pipeline.py         # End-to-end orchestration
│   │
│   ├── graph_rag/              # Graph RAG pipeline
│   │   ├── entity_extractor.py # LLM-based NER & relation extraction
│   │   ├── graph_builder.py    # Knowledge graph construction
│   │   ├── community.py        # Community detection & summarisation
│   │   ├── retriever.py        # Graph-augmented retrieval
│   │   └── pipeline.py         # End-to-end orchestration
│   │
│   ├── evaluation/             # Benchmarking
│   │   ├── metrics.py          # Precision, Recall, MRR, ROUGE
│   │   ├── benchmark.py        # Side-by-side runner
│   │   └── questions.py        # Eval question management
│   │
│   └── utils/                  # Shared utilities
│       ├── logging_setup.py
│       └── io_helpers.py
│
├── tests/                      # pytest test suite
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_graph_builder.py
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_experiment.ipynb
│   └── 03_graphrag_experiment.ipynb
│
├── scripts/                    # One-off scripts
│
├── data/                       # Data directory (gitignored contents)
│   ├── raw/                    # Raw .txt files from ArXiv extraction
│   ├── processed/              # Cleaned/chunked data
│   ├── embeddings/             # Vector store persistence
│   └── graphs/                 # Serialised knowledge graphs
│
└── results/                    # Benchmark outputs (CSVs, plots)
```

---

## Platform & Tool Decisions

| Component | Chosen Tool | Rationale | Alternatives Considered |
|-----------|------------|-----------|------------------------|
| **Language** | Python 3.10+ | Ecosystem for ML/NLP is unmatched | — |
| **Package management** | pip + pyproject.toml | Standard, simple, no extra tooling | Poetry, Conda |
| **Embeddings** | Sentence-Transformers (local) | Free, fast, no API key needed | OpenAI `text-embedding-3-small` (option) |
| **Vector store** | ChromaDB | Zero-config, file-based, good for prototyping | FAISS (available as option), Pinecone |
| **Graph store** | NetworkX | Pure Python, easy to serialise, sufficient for ~1K papers | Neo4j (option for larger scale) |
| **Community detection** | Leiden algorithm | Best quality per Microsoft GraphRAG paper | Louvain (fallback) |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, high quality | Ollama/local models (option) |
| **Orchestration** | Custom Python (no LangChain for core) | Full control, educational value | LangChain (used lightly for LLM wrappers) |
| **Evaluation** | 3-tier benchmark (LLM judge + efficiency + small golden set) | Scalable, faster iteration, still grounded by sanity checks | Full manual labeling, ROUGE-only |
| **Code quality** | Ruff + pre-commit | Fastest linter, auto-format, catches issues before commit | Black + isort + flake8 |
| **Testing** | pytest | Standard, well-supported | unittest |
| **Version control** | Git + GitHub | Team standard, PR reviews, CI-ready | — |
| **Notebooks** | Jupyter | Interactive exploration and visualisation | — |

### Why self-built over Microsoft's GraphRAG library?
- **Educational value**: Understanding each component deeply.
- **Customisability**: Tailored entity types, graph schemas, retrieval strategies.
- **Benchmarking fairness**: Comparing pipelines at the same abstraction level.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### 1. Clone & setup

```bash
git clone <repo-url>
cd UOG_RL_GraphRAG

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install in editable mode with dev tools
pip install -e ".[dev,notebooks]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (if using OpenAI)
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

### 4. Place data

Copy the paper files (`.json` or `.txt`) into `data/processed/`.

> **Note:** PDF extraction is handled by a separate team member.
> See `src/data/preprocessor.py` for the JSON schema reference if you
> need to produce structured files from raw PDFs.

### 5. Run

```bash
# Index for standard RAG (reads from data/processed/)
rag-bench index

# Build GraphRAG (entity extraction + graph + communities)
rag-bench build-graph

# Ask a question
rag-bench query "What is Proximal Policy Optimization?"

# Run benchmark
rag-bench benchmark data/eval_questions.json
```

---

## Usage

### As a library (from Python / notebooks)

```python
from src.rag.pipeline import RAGPipeline
from src.graph_rag.pipeline import GraphRAGPipeline

# Standard RAG
rag = RAGPipeline()
rag.index()
result = rag.query("What are the main RL algorithms?")
print(result.answer)

# Graph RAG
graphrag = GraphRAGPipeline()
graphrag.build()
result = graphrag.query("What are the main RL algorithms?")
print(result.answer)
```

### As a CLI

```bash
rag-bench --help
```

---

## Team Workflow (Git)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Key points:

- **Branch naming**: `feature/<name>`, `fix/<name>`, `experiment/<name>`
- **PR required** for merging to `main`
- **Pre-commit hooks** enforce consistent formatting
- **Tests must pass** before merging: `pytest`

---

## Evaluation

The project now follows a **3-tier benchmarking approach**.

### Tier 1 — Automated Sensemaking / Answer Quality (primary)

- Generate diverse evaluation questions with an LLM (persona/sensemaking style).
- Evaluate answers with **LLM-as-judge** criteria, prioritising:
      - faithfulness (groundedness)
      - comprehensiveness
      - diversity / coverage
- Run head-to-head comparisons between `rag`, `graphrag_local`, `graphrag_global`,
      `graphrag_graph_only`, and `graphrag_hybrid`.

Why: semantic evaluation scales better than ROUGE for research QA where multiple
correct phrasings and evidence paths exist.

### Tier 2 — Automated Efficiency Tracking (primary)

- Track runtime and cost proxies directly in benchmark output:
      - query latency (`latency_s`)
      - context size (`context_char_count`, `estimated_context_tokens`)
      - provider/model metadata (`llm_provider`, `llm_model`, etc.)
      - run reproducibility (`run_id`, `timestamp_utc`)

Why: GraphRAG quality gains must be weighed against latency and token/context cost.

### Tier 3 — Small Manual Golden Set (sanity check)

- Maintain a **small** manually-labeled set (recommended: 20–50 questions).
- Use exact `relevant_doc_ids` for traditional retrieval sanity metrics:
      - Precision@k
      - Recall@k
      - MRR

Why: catches obvious retrieval regressions without the cost of full manual annotation.

### Practical Guidance

- Treat LLM-judge + efficiency as the main decision signals.
- Use ROUGE as a secondary/diagnostic metric only.
- Avoid large-scale manual reference-answer authoring for the whole corpus.

Results are saved to `results/benchmark_report.csv`.

---

## Roadmap / TODOs

- [x] Project scaffolding & documentation
- [ ] Load and validate ~1,000 ArXiv paper text files
- [ ] Standard RAG: end-to-end index + query working
- [ ] GraphRAG: entity extraction on full corpus
- [ ] GraphRAG: community detection & summarisation
- [ ] Automated question generation pipeline (persona/sensemaking prompts)
- [ ] LLM-judge rubric expansion (faithfulness + comprehensiveness + diversity)
- [ ] Golden-set curation (20–50 precision/recall sanity questions)
- [ ] Full benchmark run & analysis (quality + efficiency trade-off)
- [ ] Title clustering / sub-field classification (optional)
- [ ] Results write-up & visualisation
