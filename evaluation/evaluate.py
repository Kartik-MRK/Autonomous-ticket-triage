"""
Evaluation Runner Module
===========================
Runs evaluation on a test split of processed issues.

Evaluation approach:
1. Split processed data into train (80%) and test (20%)
2. For each test issue:
   a. Run the pipeline to get classification predictions
   b. Compare against ground-truth labels (mapped from GitHub labels)
   c. Measure retrieval quality using the test issue's known labels
3. Compute aggregate metrics and print a summary report

Label Mapping:
GitHub issues often have labels like "bug", "feature-request", "enhancement".
We map these to our classification categories for evaluation.
"""

import json
import time
from typing import Optional

from modules.preprocessing import load_processed_issues
from modules.classifier import classify_ticket
from modules.retrieval import hybrid_retrieve
from modules.reranker import rerank
from evaluation.metrics import (
    classification_accuracy,
    classification_f1,
    full_classification_report,
    batch_hit_at_k,
    batch_recall_at_k,
)
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Label Mapping: GitHub Labels -> Our Categories
# ============================================
TYPE_LABEL_MAP = {
    # Bug labels
    "bug": "bug",
    "confirmed-bug": "bug",
    "testbug": "bug",
    "regression": "bug",
    "crash": "bug",
    "error": "bug",
    # Feature labels
    "feature-request": "feature",
    "feature": "feature",
    "new-feature": "feature",
    "enhancement": "improvement",
    # Improvement labels
    "improvement": "improvement",
    "polish": "improvement",
    "performance": "improvement",
    "ux": "improvement",
    "accessibility": "improvement",
    "debt": "improvement",
}

TEAM_LABEL_MAP = {
    # Frontend
    "editor": "frontend",
    "workbench": "frontend",
    "themes": "frontend",
    "keybindings": "frontend",
    "ui": "frontend",
    "css": "frontend",
    "html": "frontend",
    "layout": "frontend",
    "icon": "frontend",
    "tree-widget": "frontend",
    "quick-pick": "frontend",
    "editor-rendering": "frontend",
    # Backend
    "git": "backend",
    "debug": "backend",
    "terminal": "backend",
    "search": "backend",
    "languages": "backend",
    "typescript": "backend",
    "json": "backend",
    "python": "backend",
    "java": "backend",
    "extensions": "backend",
    "api": "backend",
    "file-io": "backend",
    # Infrastructure
    "build": "infrastructure",
    "ci": "infrastructure",
    "install": "infrastructure",
    "update": "infrastructure",
    "electron": "infrastructure",
    "performance": "infrastructure",
    "memory": "infrastructure",
    "startup": "infrastructure",
    "telemetry": "infrastructure",
}


def _extract_ground_truth(issue: dict) -> dict:
    """
    Extract ground truth classification from issue labels.

    Args:
        issue: Processed issue dictionary.

    Returns:
        Dictionary with type, severity, team (or None values if unknown).
    """
    labels = [label.lower() for label in issue.get("labels", [])]

    # Determine type
    issue_type = None
    for label in labels:
        for key, value in TYPE_LABEL_MAP.items():
            if key in label:
                issue_type = value
                break
        if issue_type:
            break

    # Determine team
    team = None
    for label in labels:
        for key, value in TEAM_LABEL_MAP.items():
            if key in label:
                team = value
                break
        if team:
            break

    return {
        "type": issue_type,
        "team": team,
        # Severity is hard to extract from labels, skip it for evaluation
    }


def run_evaluation(
    test_split_ratio: float = 0.2,
    max_test_samples: int = 50,
    processed_data_path: Optional[str] = None,
) -> dict:
    """
    Run evaluation on a test split of processed issues.

    Args:
        test_split_ratio: Fraction of data to use for testing (0.0-1.0).
        max_test_samples: Maximum number of test samples to evaluate.
        processed_data_path: Path to processed issues file.

    Returns:
        Dictionary with all evaluation results.
    """
    logger.info("=" * 60)
    logger.info("Starting Evaluation")
    logger.info("=" * 60)

    # Load processed data
    issues = load_processed_issues(processed_data_path)
    total = len(issues)

    # Split into train/test
    split_idx = int(total * (1 - test_split_ratio))
    test_issues = issues[split_idx:]

    # Limit test samples
    if len(test_issues) > max_test_samples:
        test_issues = test_issues[:max_test_samples]

    logger.info(f"Total issues: {total}, Test split: {len(test_issues)}")

    # ============================================
    # Classification Evaluation
    # ============================================
    type_true = []
    type_pred = []
    team_true = []
    team_pred = []

    # ============================================
    # Retrieval Evaluation
    # ============================================
    all_relevant_ids = []
    all_retrieved_ids = []

    start_time = time.time()

    for i, issue in enumerate(test_issues):
        logger.info(f"Evaluating issue {i + 1}/{len(test_issues)}: #{issue.get('number', 'N/A')}")

        # Extract ground truth
        ground_truth = _extract_ground_truth(issue)

        # Only evaluate if we have ground truth
        has_type_gt = ground_truth["type"] is not None
        has_team_gt = ground_truth["team"] is not None

        if not has_type_gt and not has_team_gt:
            logger.info(f"  Skipping - no ground truth labels")
            continue

        try:
            # Run classification
            prediction = classify_ticket(
                title=issue.get("clean_title", issue.get("original_title", "")),
                description=issue.get("clean_body", issue.get("original_body", "")),
                labels=issue.get("labels", []),
            )

            # Collect classification results
            if has_type_gt:
                type_true.append(ground_truth["type"])
                type_pred.append(prediction["type"])

            if has_team_gt:
                team_true.append(ground_truth["team"])
                team_pred.append(prediction["team"])

            # Run retrieval evaluation
            query = issue.get("unified_text", f"{issue.get('clean_title', '')} {issue.get('clean_body', '')}")
            retrieved = hybrid_retrieve(query[:500])  # Truncate for efficiency

            # For retrieval evaluation, consider issues with same labels as relevant
            issue_labels = set(issue.get("labels", []))
            retrieved_ids = [doc.get("id", "") for doc in retrieved]

            # Simple relevance: same label overlap
            relevant_ids = []
            for doc in retrieved:
                doc_labels = set(doc.get("metadata", {}).get("labels", "").split(", "))
                if doc_labels & issue_labels:
                    relevant_ids.append(doc.get("id", ""))

            all_relevant_ids.append(relevant_ids)
            all_retrieved_ids.append(retrieved_ids)

            logger.info(
                f"  Predicted: type={prediction['type']}, team={prediction['team']} | "
                f"  Ground truth: type={ground_truth['type']}, team={ground_truth['team']}"
            )

        except Exception as e:
            logger.error(f"  Error evaluating issue #{issue.get('number')}: {e}")
            continue

        # Rate limiting for API calls
        time.sleep(0.5)

    elapsed = time.time() - start_time

    # ============================================
    # Compute Metrics
    # ============================================
    results = {
        "num_evaluated": len(type_true),
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    # Type classification metrics
    if type_true:
        logger.info("\n--- Issue Type Classification ---")
        results["type_accuracy"] = classification_accuracy(type_true, type_pred)
        results["type_f1_weighted"] = classification_f1(type_true, type_pred, "weighted")
        results["type_f1_macro"] = classification_f1(type_true, type_pred, "macro")
        results["type_report"] = full_classification_report(type_true, type_pred)

    # Team classification metrics
    if team_true:
        logger.info("\n--- Team Routing Classification ---")
        results["team_accuracy"] = classification_accuracy(team_true, team_pred)
        results["team_f1_weighted"] = classification_f1(team_true, team_pred, "weighted")
        results["team_f1_macro"] = classification_f1(team_true, team_pred, "macro")
        results["team_report"] = full_classification_report(team_true, team_pred)

    # Retrieval metrics
    if all_relevant_ids:
        logger.info("\n--- Retrieval Metrics ---")
        for k in [1, 3, 5, 10]:
            results[f"hit_at_{k}"] = batch_hit_at_k(all_relevant_ids, all_retrieved_ids, k)
            results[f"recall_at_{k}"] = batch_recall_at_k(all_relevant_ids, all_retrieved_ids, k)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete in {elapsed:.1f}s")
    logger.info("=" * 60)

    # Save results
    results_path = settings.DATA_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Filter out non-serializable values
        serializable = {k: v for k, v in results.items() if not isinstance(v, str) or len(v) < 1000}
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results
