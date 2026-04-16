"""
Shared Evaluation Engine
========================
Common evaluation logic used by all three retrieval test modes.

Uses the ROOT pipeline modules directly — no duplication of Ollama
logic. The only parameter that varies is the retrieval function and
whether HyDE is enabled (hybrid only).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

from sklearn.metrics import f1_score, precision_score, recall_score

# ---- Use the ROOT pipeline modules directly ----
from modules.preprocessing import clean_text_regex
from modules.reranker import rerank, get_top_confidence
from modules.classifier import classify_ticket
from modules.generator import generate_response
from modules.hyde import hyde_retrieve
from config.settings import settings


# ============================================
# Metric helpers (pure functions, no Ollama)
# ============================================

def normalize_label(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def load_json_records(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    return payload


def precision_at_k(binary_relevance: List[int], k: int) -> float:
    top_k = binary_relevance[:k]
    return float(sum(top_k) / len(top_k)) if top_k else 0.0


def recall_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(sum(binary_relevance[:k]) / total_relevant)


def reciprocal_rank(binary_relevance: List[int]) -> float:
    for idx, is_relevant in enumerate(binary_relevance, start=1):
        if is_relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    top_k = binary_relevance[:k]
    dcg = sum(1.0 / math.log2(idx + 1) for idx, rel in enumerate(top_k, start=1) if rel)
    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


# ============================================
# Main evaluation loop
# ============================================

def run_evaluation(
    issues: List[Dict[str, object]],
    retrieve_fn: Callable,
    retrieval_mode: str,
    top_k: int,
    corpus_team_counter: Counter,
    known_teams: List[str],
    responses_output: Path,
    metrics_output: Path,
    hyde_enabled: bool = False,
    use_reranker: bool = True,
) -> Dict[str, object]:
    """
    Run evaluation loop using ROOT pipeline modules.

    Only `retrieve_fn` varies between tests (dense / sparse / hybrid).
    Classification and generation ALWAYS use the root modules:
      - modules.classifier.classify_ticket
      - modules.generator.generate_response
    HyDE is only triggered when `hyde_enabled=True` AND agreement < threshold.
    HyDE fires at most ONCE per query (no loops).

    Parameters
    ----------
    use_reranker : bool
        If True (default), apply cross-encoder reranking after retrieval.
        If False (baseline mode), skip reranking — use raw retrieval scores.
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    top1_hits = top3_hits = top5_hits = 0
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    response_rows: List[Dict[str, object]] = []
    hyde_trigger_count = 0

    start = time.time()

    for index, issue in enumerate(issues, start=1):
        issue_id = issue.get("id")
        query_text = str(issue.get("text", "")).strip()
        if not query_text:
            continue

        # ---- Stage 1: Preprocessing (root module) ----
        clean_query = clean_text_regex(query_text)
        title = str(issue.get("title", "")).strip() or clean_query[:140]

        # ---- Stage 2: Retrieval (varies per test) ----
        retrieved_docs = retrieve_fn(clean_query)

        # ---- Stage 3: Reranking (conditional) ----
        if use_reranker:
            reranked_docs = rerank(clean_query, retrieved_docs, top_n=max(10, top_k))
        else:
            # Baseline mode: skip reranker, use raw retrieval order
            reranked_docs = retrieved_docs[:max(10, top_k)]

        # ---- Stage 4: Classification (root module, with known_teams + RAC) ----
        classification = classify_ticket(
            title=title,
            description=clean_query[:2000],
            labels=[f"component:{issue.get('component', '')}", f"team:{issue.get('team', '')}"],
            known_teams=known_teams,
            retrieved_docs=reranked_docs[:10],  # top-10 for vote tally (Fix A+B)
        )

        # ---- Stage 4b: HyDE — triggered by classifier-retrieval AGREEMENT ----
        hyde_activated = False
        if hyde_enabled and reranked_docs:
            # Compute agreement: how many of the top-5 retrieved teams match the predicted team?
            retrieved_team_list = [
                normalize_label(
                    (doc.get("metadata", {}) if isinstance(doc, dict) else {}).get("team")
                )
                for doc in reranked_docs[:top_k]
            ]
            pred_team = normalize_label(classification.get("team"))
            matching = sum(1 for rt in retrieved_team_list if rt == pred_team)
            agreement = matching / max(len(retrieved_team_list), 1)

            if agreement < settings.HYDE_AGREEMENT_THRESHOLD:
                print(
                    f"  [HyDE] Low agreement {agreement:.2f} < {settings.HYDE_AGREEMENT_THRESHOLD} "
                    f"(pred={pred_team}, retrieved={retrieved_team_list}), activating HyDE..."
                )
                hyde_docs = hyde_retrieve(clean_query)
                if hyde_docs:
                    hyde_reranked = rerank(clean_query, hyde_docs, top_n=max(10, top_k))
                    hyde_confidence = get_top_confidence(hyde_reranked)
                    top_confidence = get_top_confidence(reranked_docs)
                    if hyde_confidence > top_confidence:
                        print(f"  [HyDE] Improved: {top_confidence:.4f} → {hyde_confidence:.4f}")
                        reranked_docs = hyde_reranked
                        # Re-classify with the improved retrieved docs
                        classification = classify_ticket(
                            title=title,
                            description=clean_query[:2000],
                            labels=[f"component:{issue.get('component', '')}", f"team:{issue.get('team', '')}"],
                            known_teams=known_teams,
                            retrieved_docs=reranked_docs[:10],  # top-10 for vote tally (Fix A+B)
                        )
                        print(f"  [HyDE] Re-classified team: {classification.get('team')}")
                    else:
                        print(f"  [HyDE] No improvement, keeping original")
                hyde_activated = True
                hyde_trigger_count += 1

        # ---- Stage 5: Generation (root module) ----
        generated = generate_response(
            title=title,
            description=clean_query[:2000],
            labels=[f"component:{issue.get('component', '')}"],
            classification=classification,
            retrieved_docs=reranked_docs[:top_k],
        )

        # ---- Evaluation: compare predicted team vs ground truth ----
        true_team = normalize_label(issue.get("team"))
        pred_team = normalize_label(classification.get("team"))

        y_true.append(true_team)
        y_pred.append(pred_team)

        top1_correct = int(pred_team == true_team)
        top1_hits += top1_correct

        # For top-3/top-5: also check if ground truth matches any retrieved doc teams
        retrieved_teams = []
        for doc in reranked_docs[:5]:
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            t = normalize_label(meta.get("team"))
            if t != "unknown" and t not in retrieved_teams:
                retrieved_teams.append(t)
        top3_correct = int(true_team in retrieved_teams[:3] or pred_team == true_team)
        top5_correct = int(true_team in retrieved_teams[:5] or pred_team == true_team)
        top3_hits += top3_correct
        top5_hits += top5_correct

        # Retrieval relevance metrics
        relevance_flags: List[int] = []
        references: List[Dict[str, object]] = []
        for doc in reranked_docs:
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            doc_team = normalize_label(metadata.get("team"))
            is_relevant = 1 if doc_team == true_team else 0
            relevance_flags.append(is_relevant)
            references.append({
                "id": doc.get("id", ""),
                "issue_number": metadata.get("issue_number", "N/A"),
                "title": metadata.get("title", "Unknown"),
                "team": metadata.get("team", "unknown"),
                "component": metadata.get("component", "unknown"),
                "score": float(doc.get("rerank_score", doc.get("score", doc.get("rrf_score", 0.0)))),
                "relevant": bool(is_relevant),
            })

        total_relevant = int(corpus_team_counter.get(true_team, 0))
        p_k = precision_at_k(relevance_flags, top_k)
        r_k = recall_at_k(relevance_flags, top_k, total_relevant)
        rr = reciprocal_rank(relevance_flags)
        ndcg = ndcg_at_k(relevance_flags, top_k, total_relevant)

        precision_scores.append(p_k)
        recall_scores.append(r_k)
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
                "predicted_team": pred_team,
                "top1_correct": bool(top1_correct),
                "top3_correct": bool(top3_correct),
                "top5_correct": bool(top5_correct),
            },
            "hyde_activated": hyde_activated,
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

        hyde_tag = " [HyDE]" if hyde_activated else ""
        print(
            f"[{index}/{len(issues)}] id={issue_id} "
            f"team_true={true_team} team_pred={pred_team} "
            f"P@{top_k}={p_k:.3f} R@{top_k}={r_k:.3f} MRR={rr:.3f}{hyde_tag}"
        )

    elapsed = time.time() - start
    total = len(y_true)
    if total == 0:
        raise RuntimeError("No issues were evaluated.")

    cls_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    cls_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    cls_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "retrieval_mode": retrieval_mode,
        "hyde_enabled": hyde_enabled,
        "reranker_used": use_reranker,
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
            "precision_at_k": float(statistics.mean(precision_scores)) if precision_scores else 0.0,
            "recall_at_k": float(statistics.mean(recall_scores)) if recall_scores else 0.0,
            "mrr": float(statistics.mean(mrr_scores)) if mrr_scores else 0.0,
            "ndcg_at_k": float(statistics.mean(ndcg_scores)) if ndcg_scores else 0.0,
        },
        "meta": {
            "num_queries": total,
            "hyde_triggers": hyde_trigger_count,
            "elapsed_seconds": round(elapsed, 2),
            "responses_file": str(responses_output),
            "metrics_file": str(metrics_output),
        },
    }

    # Save results
    responses_output.parent.mkdir(parents=True, exist_ok=True)
    with open(responses_output, "w", encoding="utf-8") as handle:
        json.dump(response_rows, handle, indent=2, ensure_ascii=False)
    with open(metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(response_rows)} query responses → {responses_output}")
    print(f"Saved evaluation metrics → {metrics_output}")

    return metrics
