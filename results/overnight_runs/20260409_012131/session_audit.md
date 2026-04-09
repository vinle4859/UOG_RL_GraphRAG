# Overnight Benchmark Audit

- Status: SUCCESS
- Started: 2026-04-09 01:21:32 +07:00
- Finished: 2026-04-09 03:02:51 +07:00
- Duration: 01:41:18.6886032
- Model: gemma4:e2b
- Question set: data/eval_questions_alt.json
- Community summary limit: 50
- Smoke question count: 3
- Judge workers: 2
- Benchmark retries: 3

## Artifacts
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\overnight_benchmark.log (3.1 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\session_audit.md (this file)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_smoke.csv (104.4 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_smoke.md (2.6 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_smoke.jsonl (110.4 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_full.csv (925.6 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_full.md (3 KB)
- D:\GW_UNIVERSITY\ResearchLab\UOG_RL_GraphRAG\results\overnight_runs\20260409_012131\benchmark_full.jsonl (974.6 KB)

## Failure
- None

## Git Status
```text
M  README.md
M  data/eval_questions_alt.json
 M pyproject.toml
A  results/benchmark_consolidated_report_2026-03-19.md
A  results/benchmark_report_42b2a983_2026-04-02.md
A  results/benchmark_runs/42b2a983-d272-45e2-9abc-5c3e573b50c7.jsonl
A  results/benchmark_runs/ed4c4752-3b31-4602-a13c-f6d9bc26cc30.jsonl
A  results/benchmark_runs/f08e0a75-4094-436b-99e2-e6b84066d8ac.jsonl
M  results/escalation_analysis.png
AM results/graph_visualizer/knowledge_graph.html
M  scripts/inspect_failed_chunks.ipynb
MM src/cli.py
 M src/config.py
MM src/evaluation/benchmark.py
 M src/evaluation/metrics.py
 M src/graph_rag/community.py
 M src/graph_rag/pipeline.py
 M src/graph_rag/retriever.py
 M src/rag/embedder.py
 M src/rag/generator.py
 M src/rag/pipeline.py
 M src/utils/logging_setup.py
A  src/visualization/__init__.py
AM src/visualization/graphrag_artifacts.py
MM tests/test_benchmark.py
 M tests/test_graph_retriever.py
 M tests/test_rag_smoke.py
?? results/benchmark_runs/overnight_smoke_tmp.jsonl
?? results/benchmark_runs/smoke_20260409_004758.jsonl
?? results/benchmark_runs/smoke_opt_20260409_010839.jsonl
?? results/overnight_runs/
?? results/smoke_20260409_004758.md
?? results/smoke_opt_20260409_010839.md
?? scripts/nightly_full_benchmark.ps1
?? src/evaluation/preflight.py
?? src/evaluation/reporting.py
?? src/visualization/graph_visualizer.py
?? tests/test_community_store.py
?? tests/test_graph_visualizer.py
?? tests/test_preflight.py
```

## Git Diff Stat
```text
 pyproject.toml                                |   4 +-
 results/graph_visualizer/knowledge_graph.html |  50 +--
 src/cli.py                                    | 339 +++++++++------
 src/config.py                                 |  26 ++
 src/evaluation/benchmark.py                   | 591 ++++++++++++++++----------
 src/evaluation/metrics.py                     |  36 +-
 src/graph_rag/community.py                    |  75 +++-
 src/graph_rag/pipeline.py                     | 324 +++++++-------
 src/graph_rag/retriever.py                    | 163 ++++---
 src/rag/embedder.py                           |  11 +-
 src/rag/generator.py                          |  21 +-
 src/rag/pipeline.py                           |   3 +-
 src/utils/logging_setup.py                    |   9 +
 src/visualization/graphrag_artifacts.py       | 118 ++---
 tests/test_benchmark.py                       | 150 +++++--
 tests/test_graph_retriever.py                 |  51 +++
 tests/test_rag_smoke.py                       |   3 +
 17 files changed, 1298 insertions(+), 676 deletions(-)
```
