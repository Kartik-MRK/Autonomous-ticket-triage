"""
Dense-Only Retrieval Evaluation (BASELINE)
============================================
Uses ONLY dense (vector) retrieval from root modules.
NO reranker, NO HyDE. Serves as a baseline for comparison
against the full hybrid pipeline.

Retrieves top-40 candidates using raw cosine similarity scores.
Classification uses fine-grained known_teams + RAC.

Uses test_processed.json (100 issues) → saves results to results/.
Usage: python tests/dense_only_retrival/run_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TESTS_DIR = THIS_DIR.parent
ROOT_DIR = TESTS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from modules.retrieval import dense_retrieve
from modules.vector_store import get_all_documents, get_collection_stats
from eval_engine import load_json_records, normalize_label, unique_preserve_order, run_evaluation

# Paths
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "test_processed.json"
RESULTS_DIR = THIS_DIR / "results"
DEFAULT_RESPONSES = RESULTS_DIR / "query_responses.json"
DEFAULT_METRICS = RESULTS_DIR / "evaluation_metrics.json"


def _retrieve_dense(query: str) -> list:
    """Dense-only retrieval using root modules.retrieval (top-40, no reranker)."""
    return dense_retrieve(query, top_k=40)


def main() -> None:
    input_path = DEFAULT_INPUT
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run 'python main.py build-index' first to create test_processed.json")
        sys.exit(1)

    issues = load_json_records(input_path)
    print(f"Loaded {len(issues)} test issues from {input_path}")

    stats = get_collection_stats()
    if int(stats.get("count", 0)) <= 0:
        print("ChromaDB is empty. Run 'python main.py build-index' first.")
        sys.exit(1)

    all_docs = get_all_documents()
    corpus_metadatas = all_docs.get("metadatas", []) or []
    corpus_team_counter = Counter(
        normalize_label(meta.get("team"))
        for meta in corpus_metadatas if isinstance(meta, dict)
    )
    known_teams = unique_preserve_order(
        [normalize_label(item.get("team")) for item in issues if item.get("team")]
        + list(corpus_team_counter.keys())
    )

    metrics = run_evaluation(
        issues=issues,
        retrieve_fn=_retrieve_dense,
        retrieval_mode="dense_only",
        top_k=5,
        corpus_team_counter=corpus_team_counter,
        known_teams=known_teams,
        responses_output=DEFAULT_RESPONSES,
        metrics_output=DEFAULT_METRICS,
        hyde_enabled=False,   # NO HyDE for baseline
        use_reranker=False,   # NO reranker for baseline
    )

    print("\n" + "=" * 60)
    print("DENSE-ONLY EVALUATION COMPLETE")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
