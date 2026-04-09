# Benchmarking Strategy (RAG vs GraphRAG)

This document defines the project’s recommended evaluation methodology.

## Why the approach changed

Manual full-corpus annotation does not scale well for research QA:

- Many questions have multiple valid supporting papers/chunks.
- Strict exact-document labeling can undercount true positives.
- ROUGE can penalize correct answers with different wording.

For this project, we optimise for scalable iteration and production-relevant
trade-offs (quality vs latency/cost), while keeping a small manual sanity check.

## Three-Tier Evaluation

### Tier 1: Automated sensemaking quality (primary)

- Use an LLM to generate a broad question set from paper abstracts/full text.
- Compare RAG and GraphRAG modes head-to-head.
- Judge answers semantically using LLM-as-judge criteria:
  - faithfulness (groundedness)
  - comprehensiveness
  - diversity/coverage
  - directness
  - empowerment

Current status:
- Implemented: all five LLM-judge metrics with rubric-driven prompts.
- Implemented: optional judge rationale capture for human review.

### Tier 2: Automated efficiency tracking (primary)

Every benchmark row should include:

- latency (`latency_s`)
- context size (`context_char_count`, `estimated_context_tokens`)
- run metadata (`run_id`, `timestamp_utc`)
- model/provider metadata (`llm_provider`, `llm_model`, embedding settings)
- retrieval traceability (`retrieved_chunk_ids`, chunk scores, hits@k)

Human-review exports additionally include top-k chunk text + metadata.

This enables quality-cost trade-off analysis and reproducibility.

Implementation note:
- Efficiency tracking is now configurable in the benchmark CLI.
- Use `--no-efficiency-tracking` when you want quality-only runs.

### Tier 3: Small manual golden set (sanity)

- Maintain only a 20–50 question manually labeled set.
- Use it for retrieval sanity metrics:
  - Precision@k
  - Recall@k
  - MRR

This verifies retrieval integrity without full manual bottlenecks.

## Recommended workflow

1. Generate or refresh automated question set.
2. Run benchmark across all pipelines/modes.
3. Review `results/benchmark_review.md` for question-by-question evidence.
4. Compare quality metrics first, then efficiency metrics.
5. Run golden-set retrieval sanity check before major releases.

## CLI examples

```bash
# Full benchmark with default GraphRAG modes
rag-bench benchmark data/eval_questions.json

# Restrict to selected GraphRAG modes
rag-bench benchmark data/eval_questions.json --graphrag-mode local --graphrag-mode graph_only

# Enable faithfulness (LLM-as-judge)
rag-bench benchmark data/eval_questions.json --faithfulness

# Default run already enables all judge metrics + rationale collection
rag-bench benchmark data/eval_questions.json

# Fast retrieval-focused run (disable judge calls)
rag-bench benchmark data/eval_questions.json --no-faithfulness --no-quality-judges

# Quality-focused run with lighter benchmark bookkeeping
rag-bench benchmark data/eval_questions.json --no-efficiency-tracking

# Outputs written per run
# - results/benchmark_report.csv
# - results/benchmark_review.csv
# - results/benchmark_review.md
```
