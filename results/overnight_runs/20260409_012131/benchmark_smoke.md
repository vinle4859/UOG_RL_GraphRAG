# Benchmark Report: Run 86e73103

## 1. Run Configuration

- Date generated: 2026-04-09
- Run ID: 86e73103-8795-4a16-90fb-c745b0ed41e8
- Run timestamp (UTC): 2026-04-08T18:41:17.072083+00:00
- Question file used: data/eval_questions_alt.json
- Top-k: 5
- Pipelines evaluated: graphrag_global, graphrag_graph_only, graphrag_hybrid, graphrag_local, rag
- GraphRAG modes requested: local, global, graph_only, hybrid
- Benchmark model override: gemma4:e2b

## 2. Tier Coverage Check

- Tier 1 (LLM judges): Available via faithfulness, comprehensiveness, diversity, directness, empowerment
- Tier 2 (efficiency tracking): Available and populated
- Tier 3 (retrieval sanity metrics): Available and informative

## 3. Aggregate Results

|Pipeline|Rows|rouge1|rougeL|faithfulness|latency_s|est_context_tokens|fallback_rate|empty_answer_rate|
|---|---|---|---|---|---|---|---|---|
|graphrag_global|3|0.2797|0.2167|0.3333|18.6747|2366.0000|33.3%|0.0%|
|graphrag_graph_only|3|0.2458|0.1891|0.6667|84.5991|172.6667|0.0%|0.0%|
|graphrag_hybrid|3|0.2091|0.1470|0.3333|12.0579|38108.0000|0.0%|0.0%|
|graphrag_local|3|0.2025|0.1421|0.3000|15.1458|35741.0000|33.3%|0.0%|
|rag|3|0.2270|0.1466|1.0000|33.3666|2562.6667|0.0%|0.0%|

## 4. Reliability and Failure Summary

- Highest fallback rate: `graphrag_local` at 33.3%.
- Lowest average latency: `graphrag_hybrid` at 12.0579 s.
- Largest average context: `graphrag_hybrid` at 38108.0000 tokens.

## 5. Best and Worst Examples

### Best Example

- Question: What is the relationship between the Sinkhorn algorithm and the parabolic Monge-Ampère (PMA) PDE in the context of Wasserstein mirror gradient flows?
- Pipeline: graphrag_graph_only
- rouge1: 0.4318
- latency_s: 32.0543
- Answer snippet: The provided context does not contain information regarding the relationship between the Sinkhorn algorithm and the parabolic Monge-Ampère (PMA) PDE in the context of Wasserstein mirror gradient flows.

### Worst Example

- Question: How do the ASC framework and the COMSPLIT design respectively address the issue of unreliable or scarce data at the edge?
- Pipeline: graphrag_graph_only
- rouge1: 0.0519
- latency_s: 8.8884
- Answer snippet: I do not have the context required to answer your question.

## 6. Recommendations

- Keep `graphrag_global` as the current lexical-quality baseline; it has the strongest average rouge1 in this run.
- `rag` is currently strongest on faithfulness and is worth re-checking against latency and fallback rates.
- Investigate `graphrag_local` before using it in overnight runs; fallback-style answers appear in 33.3% of rows.
