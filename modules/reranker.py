"""
Cross-Encoder Reranking Module
=================================
Uses the cross-encoder/ms-marco-MiniLM-L-12-v2 model to rerank
retrieval candidates for maximum relevance.

Cross-encoders process (query, document) pairs jointly, providing
more accurate relevance scores than bi-encoder similarity, but at
higher computational cost. This is why we use it as a second stage
after initial retrieval.

Key Design:
- Model loaded lazily (singleton)
- Accepts pre-retrieved candidates from hybrid retrieval
- Returns top-N reranked results
- Preserves all original metadata through the reranking

References:
- https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from typing import Optional

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Lazy model loading (singleton)
# ============================================
_reranker_model = None


def _get_reranker():
    """Lazily load the cross-encoder reranking model."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL)
        logger.info("Reranker model loaded successfully")
    return _reranker_model


def rerank(
    query: str,
    candidates: list[dict],
    top_n: Optional[int] = None,
    document_key: str = "document",
) -> list[dict]:
    """
    Rerank retrieval candidates using the cross-encoder model.

    Takes (query, candidate_document) pairs, scores them with the
    cross-encoder, and returns the top-N candidates sorted by
    relevance score.

    Args:
        query: The original query text.
        candidates: List of candidate dictionaries from retrieval.
                    Each must have a 'document' key (or custom key).
        top_n: Number of top results to return. Defaults to settings.RERANK_TOP_N.
        document_key: Key in candidate dict containing the document text.

    Returns:
        List of top-N reranked candidate dictionaries with added 'rerank_score'.
    """
    if top_n is None:
        top_n = settings.RERANK_TOP_N

    if not candidates:
        logger.warning("No candidates to rerank")
        return []

    model = _get_reranker()

    # Build (query, document) pairs for cross-encoder
    pairs = []
    for candidate in candidates:
        doc_text = candidate.get(document_key, "")
        pairs.append([query, doc_text])

    logger.info(f"Reranking {len(pairs)} candidates...")

    # Score all pairs
    scores = model.predict(pairs)

    # Attach scores to candidates
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        reranked = candidate.copy()
        reranked["rerank_score"] = float(scores[i])
        scored_candidates.append(reranked)

    # Sort by rerank score (descending - higher is more relevant)
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top-N
    top_results = scored_candidates[:top_n]

    logger.info(
        f"Reranking complete: returned top {len(top_results)} of {len(candidates)} candidates "
        f"(best score: {top_results[0]['rerank_score']:.4f}, "
        f"worst score: {top_results[-1]['rerank_score']:.4f})"
    )

    return top_results
