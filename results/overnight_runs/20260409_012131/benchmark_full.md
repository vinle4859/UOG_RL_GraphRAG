# Benchmark Report: Run d6cb1ca7

## 1. Run Configuration

- Date generated: 2026-04-09
- Run ID: d6cb1ca7-d5d8-4a08-9085-903c8cdaa16a
- Run timestamp (UTC): 2026-04-08T18:53:12.871381+00:00
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
|graphrag_global|24|0.2336|0.1560|0.1250|19.9809|2374.2083|16.7%|0.0%|
|graphrag_graph_only|24|0.2247|0.1490|0.6667|24.7161|296.9583|8.3%|0.0%|
|graphrag_hybrid|24|0.1558|0.0976|0.4083|15.7506|47241.4583|0.0%|0.0%|
|graphrag_local|24|0.2179|0.1499|0.3333|15.9788|44866.2500|4.2%|0.0%|
|rag|24|0.2886|0.1931|0.7708|27.8766|2553.4583|4.2%|0.0%|

## 4. Reliability and Failure Summary

- Highest fallback rate: `graphrag_global` at 16.7%.
- Lowest average latency: `graphrag_hybrid` at 15.7506 s.
- Largest average context: `graphrag_hybrid` at 47241.4583 tokens.

## 5. Best and Worst Examples

### Best Example

- Question: What types of quantitative metrics were used to evaluate the similarity between DeepFoqus-Accelerate reconstructions and standard-of-care MRI images?
- Pipeline: rag
- rouge1: 0.4673
- latency_s: 24.4866
- Answer snippet: The quantitative metrics used to evaluate the similarity between DeepFoqus-Accelerate reconstructions and standard-of-care (SOC) images include: * Structural Similarity Index Measure (SSIM) [2509.07193__chunk_0, 2509.07193__chunk_6] * Pe...

### Worst Example

- Question: Why do end-to-end speech translation systems i   they lack as many dedicated training datasets as standalone ASR and MT tasks, and their encoder must simultaneously perform complex acoustic modeling and semantic encoding. To mitigate this, researchers utilized large pre-trained components (a Wav2Vec 2.0 encoder and an mBART decoder) combined with LNA finetuning and custom coupling adapters to leverage powerful external self-supervised representations.
- Pipeline: rag
- rouge1: 0.0000
- latency_s: 44.5861
- Answer snippet: The provided context discusses the shift from cascaded Automatic Speech Recognition (ASR) and Machine Translation (MT) systems to end-to-end (E2E) models. Regarding the specific points raised in the question: * **E2E vs. Cascaded Systems...

## 6. Recommendations

- Keep `rag` as the current lexical-quality baseline; it has the strongest average rouge1 in this run.
- `rag` is currently strongest on faithfulness and is worth re-checking against latency and fallback rates.
