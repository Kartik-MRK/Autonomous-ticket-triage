"""
Evaluation Metrics Module
============================
Implements evaluation metrics for the triage system:

Classification Metrics (via scikit-learn):
- Accuracy
- F1-score (weighted, macro, micro)
- Classification report

Retrieval Metrics (custom):
- Hit@K: Whether at least one relevant document appears in top-K
- Recall@K: Fraction of relevant documents found in top-K
- Mean Reciprocal Rank (MRR)

These metrics can be run independently on a test split
to evaluate system performance.
"""

import numpy as np
from typing import Union

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from utils.logger import get_logger  # noqa: uses config.settings via logger

logger = get_logger(__name__)


# ============================================
# Classification Metrics
# ============================================

def classification_accuracy(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    score = accuracy_score(y_true, y_pred)
    logger.info(f"Classification accuracy: {score:.4f}")
    return score


def classification_f1(
    y_true: list[str],
    y_pred: list[str],
    average: str = "weighted",
) -> float:
    """
    Compute F1-score for classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method - 'weighted', 'macro', 'micro'.

    Returns:
        F1 score between 0 and 1.
    """
    score = f1_score(y_true, y_pred, average=average, zero_division=0)
    logger.info(f"Classification F1 ({average}): {score:.4f}")
    return score


def classification_precision(
    y_true: list[str],
    y_pred: list[str],
    average: str = "weighted",
) -> float:
    """Compute precision score."""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def classification_recall(
    y_true: list[str],
    y_pred: list[str],
    average: str = "weighted",
) -> float:
    """Compute recall score."""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def full_classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = None,
) -> str:
    """
    Generate a full classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names.

    Returns:
        Formatted classification report string.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    logger.info(f"Classification Report:\n{report}")
    return report


# ============================================
# Retrieval Metrics
# ============================================

def hit_at_k(
    relevant_ids: Union[list[str], set[str]],
    retrieved_ids: list[str],
    k: int = 5,
) -> float:
    """
    Compute Hit@K: whether at least one relevant document
    appears in the top-K retrieved results.

    Args:
        relevant_ids: Set of relevant document IDs.
        retrieved_ids: Ordered list of retrieved document IDs.
        k: Number of top results to consider.

    Returns:
        1.0 if hit, 0.0 if miss.
    """
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    hit = 1.0 if any(doc_id in relevant_set for doc_id in top_k) else 0.0
    return hit


def recall_at_k(
    relevant_ids: Union[list[str], set[str]],
    retrieved_ids: list[str],
    k: int = 5,
) -> float:
    """
    Compute Recall@K: fraction of relevant documents found in top-K.

    Args:
        relevant_ids: Set of relevant document IDs.
        retrieved_ids: Ordered list of retrieved document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0 and 1.
    """
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_ids[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return found / len(relevant_set)


def mean_reciprocal_rank(
    relevant_ids: Union[list[str], set[str]],
    retrieved_ids: list[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR): 1/rank of the first relevant document.

    Args:
        relevant_ids: Set of relevant document IDs.
        retrieved_ids: Ordered list of retrieved document IDs.

    Returns:
        MRR score between 0 and 1.
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def batch_hit_at_k(
    all_relevant: list[list[str]],
    all_retrieved: list[list[str]],
    k: int = 5,
) -> float:
    """
    Compute average Hit@K across multiple queries.

    Args:
        all_relevant: List of relevant ID sets for each query.
        all_retrieved: List of retrieved ID lists for each query.
        k: Number of top results to consider.

    Returns:
        Average Hit@K score.
    """
    if not all_relevant:
        return 0.0

    hits = [hit_at_k(rel, ret, k) for rel, ret in zip(all_relevant, all_retrieved)]
    avg = np.mean(hits)
    logger.info(f"Average Hit@{k}: {avg:.4f} ({sum(hits):.0f}/{len(hits)} queries)")
    return float(avg)


def batch_recall_at_k(
    all_relevant: list[list[str]],
    all_retrieved: list[list[str]],
    k: int = 5,
) -> float:
    """
    Compute average Recall@K across multiple queries.

    Args:
        all_relevant: List of relevant ID sets for each query.
        all_retrieved: List of retrieved ID lists for each query.
        k: Number of top results to consider.

    Returns:
        Average Recall@K score.
    """
    if not all_relevant:
        return 0.0

    recalls = [recall_at_k(rel, ret, k) for rel, ret in zip(all_relevant, all_retrieved)]
    avg = np.mean(recalls)
    logger.info(f"Average Recall@{k}: {avg:.4f}")
    return float(avg)
