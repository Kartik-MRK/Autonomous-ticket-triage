"""Run root retrieval pipeline with Ollama reasoning and store responses + metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import f1_score, precision_score, recall_score


ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from modules.preprocessing import clean_text_regex
from modules.retrieval import hybrid_retrieve
from modules.reranker import rerank
from modules.vector_store import get_all_documents, get_collection_stats
from ollama_reasoning import classify_issue_with_ollama, generate_ticket_response_with_ollama


DEFAULT_INPUT_FILE = TESTS_DIR / "data" / "preprocessed.json"
DEFAULT_RESPONSES_FILE = TESTS_DIR / "data" / "query_responses.json"
DEFAULT_METRICS_FILE = TESTS_DIR / "data" / "evaluation_metrics.json"


def normalize_label(value: object) -> str:
    """Normalize label-like fields for consistent comparisons."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def load_json_records(path: Path) -> List[Dict[str, object]]:
    """Load JSON list records from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    return payload


def precision_at_k(binary_relevance: List[int], k: int) -> float:
    """Precision@K with binary relevance sequence."""
    top_k = binary_relevance[:k]
    if not top_k:
        return 0.0
    return float(sum(top_k) / len(top_k))


def recall_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    """Recall@K with denominator from corpus relevant count."""
    if total_relevant <= 0:
        return 0.0
    return float(sum(binary_relevance[:k]) / total_relevant)


def reciprocal_rank(binary_relevance: List[int]) -> float:
    """Reciprocal rank of first relevant hit."""
    for idx, is_relevant in enumerate(binary_relevance, start=1):
        if is_relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    """nDCG@K for binary relevance labels."""
    top_k = binary_relevance[:k]

    dcg = 0.0
    for idx, rel in enumerate(top_k, start=1):
        if rel:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0

    idcg = 0.0
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def unique_preserve_order(values: List[str]) -> List[str]:
    """Deduplicate values while preserving order."""
    seen = set()
    output: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def candidate_teams_from_retrieval(
    reranked_docs: List[Dict[str, object]],
    known_teams: List[str],
    limit: int = 10,
) -> List[str]:
    """Build candidate team list from retrieved docs + fallback pool."""
    teams: List[str] = []

    for doc in reranked_docs:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        team = normalize_label(metadata.get("team"))
        if team != "unknown":
            teams.append(team)

    teams = unique_preserve_order(teams)
    for team in known_teams:
        if len(teams) >= limit:
            break
        if team not in teams:
            teams.append(team)

    return teams[:limit]


def evaluate_with_ollama(
    issues: List[Dict[str, object]],
    top_k: int,
    responses_output: Path,
    metrics_output: Path,
) -> Dict[str, object]:
    """Run retrieval+rerank+ollama reasoning and save outputs."""
    stats = get_collection_stats()
    chroma_count = int(stats.get("count", 0))
    if chroma_count <= 0:
        raise RuntimeError(
            "Chroma collection is empty. Build the index first (python main.py build-bugzilla-index --reset)."
        )

    all_docs = get_all_documents()
    corpus_metadatas = all_docs.get("metadatas", []) or []
    corpus_team_counter = Counter(
        normalize_label(meta.get("team"))
        for meta in corpus_metadatas
        if isinstance(meta, dict)
    )

    known_teams = unique_preserve_order(
        [normalize_label(item.get("team")) for item in issues if item.get("team")]
        + list(corpus_team_counter.keys())
    )

    y_true: List[str] = []
    y_pred: List[str] = []

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    precision_at_k_scores: List[float] = []
    recall_at_k_scores: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []

    response_rows: List[Dict[str, object]] = []

    start = time.time()

    for index, issue in enumerate(issues, start=1):
        issue_id = issue.get("id")
        query_text = str(issue.get("text", "")).strip()
        if not query_text:
            continue

        clean_query = clean_text_regex(query_text)
        retrieved_docs = hybrid_retrieve(clean_query, final_top_k=max(20, top_k * 3))
        reranked_docs = rerank(clean_query, retrieved_docs, top_n=max(10, top_k))

        candidates = candidate_teams_from_retrieval(reranked_docs, known_teams, limit=10)
        classification = classify_issue_with_ollama(
            query_text=clean_query,
            component_hint=str(issue.get("component", "")),
            candidate_teams=candidates,
        )

        generated = generate_ticket_response_with_ollama(
            query_text=clean_query,
            classification=classification,
            retrieved_docs=reranked_docs[:top_k],
        )

        true_team = normalize_label(issue.get("team"))
        ranked_teams = [normalize_label(team) for team in classification.get("top_teams", [])]
        ranked_teams = [team for team in ranked_teams if team]
        if not ranked_teams:
            ranked_teams = [candidates[0] if candidates else "unknown"]

        pred_team = ranked_teams[0]
        y_true.append(true_team)
        y_pred.append(pred_team)

        top1_correct = int(pred_team == true_team)
        top3_correct = int(true_team in ranked_teams[:3])
        top5_correct = int(true_team in ranked_teams[:5])

        top1_hits += top1_correct
        top3_hits += top3_correct
        top5_hits += top5_correct

        relevance_flags: List[int] = []
        references: List[Dict[str, object]] = []

        for doc in reranked_docs:
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            doc_team = normalize_label(metadata.get("team"))
            is_relevant = 1 if doc_team == true_team else 0
            relevance_flags.append(is_relevant)

            references.append(
                {
                    "id": doc.get("id", ""),
                    "issue_number": metadata.get("issue_number", "N/A"),
                    "title": metadata.get("title", "Unknown"),
                    "team": metadata.get("team", "unknown"),
                    "component": metadata.get("component", "unknown"),
                    "score": float(doc.get("rerank_score", doc.get("rrf_score", 0.0))),
                    "relevant": bool(is_relevant),
                }
            )

        total_relevant_for_team = int(corpus_team_counter.get(true_team, 0))
        p_k = precision_at_k(relevance_flags, top_k)
        r_k = recall_at_k(relevance_flags, top_k, total_relevant_for_team)
        rr = reciprocal_rank(relevance_flags)
        ndcg = ndcg_at_k(relevance_flags, top_k, total_relevant_for_team)

        precision_at_k_scores.append(p_k)
        recall_at_k_scores.append(r_k)
        mrr_scores.append(rr)
        ndcg_scores.append(ndcg)

        row = {
            "id": issue_id,
            "query": clean_query,
            "ground_truth": {
                "team": true_team,
                "component": issue.get("component", ""),
                "assignee": issue.get("assignee", ""),
                "solution": issue.get("solution", ""),
            },
            "classification": {
                "type": classification.get("type", "bug"),
                "severity": classification.get("severity", "medium"),
                "top_teams": ranked_teams,
                "predicted_team": pred_team,
                "rationale": classification.get("rationale", ""),
                "top1_correct": bool(top1_correct),
                "top3_correct": bool(top3_correct),
                "top5_correct": bool(top5_correct),
            },
            "retrieved_references": references[:top_k],
            "generated_response": generated,
            "query_metrics": {
                f"precision_at_{top_k}": p_k,
                f"recall_at_{top_k}": r_k,
                "mrr": rr,
                f"ndcg_at_{top_k}": ndcg,
            },
        }
        response_rows.append(row)

        print(
            f"[{index}/{len(issues)}] id={issue_id} "
            f"team_true={true_team} team_pred={pred_team} "
            f"P@{top_k}={p_k:.3f} R@{top_k}={r_k:.3f} MRR={rr:.3f}"
        )

    elapsed = time.time() - start
    total = len(y_true)
    if total == 0:
        raise RuntimeError("No issues were evaluated. Check tests/data/preprocessed.json content.")

    cls_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    cls_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    cls_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "routing_classification": {
            "top1_accuracy": float(top1_hits / total),
            "top3_accuracy": float(top3_hits / total),
            "top5_accuracy": float(top5_hits / total),
            "precision": cls_precision,
            "recall": cls_recall,
            "f1_score": cls_f1,
        },
        "retrieval_rag": {
            "k": top_k,
            "recall_at_k": float(statistics.mean(recall_at_k_scores)) if recall_at_k_scores else 0.0,
            "precision_at_k": float(statistics.mean(precision_at_k_scores)) if precision_at_k_scores else 0.0,
            "mrr": float(statistics.mean(mrr_scores)) if mrr_scores else 0.0,
            "ndcg_at_k": float(statistics.mean(ndcg_scores)) if ndcg_scores else 0.0,
        },
        "meta": {
            "num_queries": total,
            "elapsed_seconds": round(elapsed, 2),
            "responses_file": str(responses_output),
            "metrics_file": str(metrics_output),
        },
    }

    responses_output.parent.mkdir(parents=True, exist_ok=True)
    with open(responses_output, "w", encoding="utf-8") as handle:
        json.dump(response_rows, handle, indent=2, ensure_ascii=False)

    with open(metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(response_rows)} query responses to {responses_output}")
    print(f"Saved evaluation metrics to {metrics_output}")

    return metrics


def parse_args() -> argparse.Namespace:
    """CLI args for Ollama evaluation run."""
    parser = argparse.ArgumentParser(
        description="Run root retrieval/rerank pipeline with Ollama reasoning on tests data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input preprocessed JSON path.",
    )
    parser.add_argument(
        "--responses-output",
        type=Path,
        default=DEFAULT_RESPONSES_FILE,
        help="Output JSON path for query-response rows.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help="Output JSON path for aggregate metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit for number of issues to evaluate (0 means all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="K value for retrieval metrics (Precision@K, Recall@K, nDCG@K).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running Ollama-based triage evaluation."""
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    issues = load_json_records(args.input)
    if args.max_samples > 0:
        issues = issues[: args.max_samples]

    metrics = evaluate_with_ollama(
        issues=issues,
        top_k=args.top_k,
        responses_output=args.responses_output,
        metrics_output=args.metrics_output,
    )

    print("Run complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
