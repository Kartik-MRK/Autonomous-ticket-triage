# Test Results Report (150 Cases)

This file is a standalone summary of the latest large-run evaluation for the autonomous ticket triage system.

It explains:
- what was measured,
- the exact numbers achieved,
- how those numbers were produced,
- and what those numbers mean for the project.

## 1. Why This Evaluation Matters

The project is designed to classify and route tickets using retrieval-grounded reasoning.

This run evaluates two core capabilities together:
- Routing quality: whether tickets are assigned to the correct team.
- Retrieval quality: whether RAG retrieves relevant historical Bugzilla references.

## 2. Run Snapshot

- Evaluation size: 150 test queries
- Runtime: 3037.18 seconds (~50.62 minutes)
- Average latency: ~20.25 seconds per query
- Output artifacts:
  - `tests/data/query_responses.json`
  - `tests/data/evaluation_metrics.json`

Sample completion log line:

`[150/150] id=2039 team_true=css parsing and computation team_pred=css parsing and computation P@5=0.600 R@5=0.057 MRR=0.500`

## 3. Achieved Metrics

### Routing Classification Metrics

| Metric | Value | Percentage |
|---|---:|---:|
| Top-1 Accuracy | 0.6066666667 | 60.67% |
| Top-3 Accuracy | 0.6266666667 | 62.67% |
| Top-5 Accuracy | 0.6533333333 | 65.33% |
| Precision (weighted) | 0.5618728124 | 56.19% |
| Recall (weighted) | 0.6066666667 | 60.67% |
| F1 Score (weighted) | 0.5666179643 | 56.66% |

### Retrieval + RAG Metrics (k=5)

| Metric | Value | Percentage |
|---|---:|---:|
| Recall@5 | 0.0110714099 | 1.11% |
| Precision@5 | 0.2293333333 | 22.93% |
| MRR | 0.3063783069 | 30.64% |
| nDCG@5 | 0.2258938565 | 22.59% |

## 4. How We Arrived At These Results

The evaluation uses the same end-to-end architecture as the project workflow:

1. Input ticket text is cleaned.
2. Hybrid retrieval runs over indexed Bugzilla data:
   - dense retrieval (embeddings),
   - sparse retrieval (BM25),
   - fusion with Reciprocal Rank Fusion (RRF).
3. Cross-encoder reranking improves ranking quality.
4. Ollama-based reasoning predicts type/severity and ranked teams.
5. Generated response uses retrieved context.
6. Metrics are aggregated over all 150 queries.

Primary command used:

`python tests/run_ollama_pipeline_eval.py`

## 5. What The Numbers Indicate

### Strong signals

- The system is useful for first-pass triage.
- Top-1 routing of 60.67% shows a majority of tickets are routed correctly on first prediction.
- Top-5 routing of 65.33% shows correct teams are frequently present in candidate rankings.
- MRR of 0.306 suggests relevant retrieval hits often appear relatively early.

### Bottleneck

- Recall@5 is low at 1.11%, indicating many relevant references are missed in top-5 retrieval.
- Precision@5 at 22.93% indicates useful references are present, but retrieval consistency needs improvement.

## 6. Project-Level Interpretation

These results validate that the architecture is functional and produces meaningful triage behavior.

They also show retrieval quality is currently the main constraint on stronger end-to-end performance.

In practical terms:
- The system is ready for assisted triage workflows.
- It should not be treated as fully autonomous routing without review.

## 7. Recommended Next Focus

1. Improve retrieval recall (query expansion, metadata-aware retrieval, larger pre-rerank pools).
2. Tune reranking and top-k settings.
3. Use Bugzilla component/team metadata more explicitly in retrieval.
4. Keep this 150-case benchmark fixed and re-run after each change.
