"""
Cross-Encoder Reranking Module
=================================
Uses BAAI/bge-reranker-base cross-encoder for reranking retrieval
candidates. Provides confidence scoring for HyDE decision.
"""

from typing import Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

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

    Returns top-N candidates sorted by relevance score, each with
    an added 'rerank_score' field.
    """
    if top_n is None:
        top_n = settings.RERANK_TOP_N

    if not candidates:
        logger.warning("No candidates to rerank")
        return []

    model = _get_reranker()

    pairs = [[query, candidate.get(document_key, "")] for candidate in candidates]
    logger.info(f"Reranking {len(pairs)} candidates...")

    scores = model.predict(pairs)

    scored_candidates = []
    for i, candidate in enumerate(candidates):
        reranked = candidate.copy()
        reranked["rerank_score"] = float(scores[i])
        scored_candidates.append(reranked)

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_results = scored_candidates[:top_n]

    logger.info(
        f"Reranking complete: top {len(top_results)} of {len(candidates)} candidates "
        f"(best: {top_results[0]['rerank_score']:.4f}, worst: {top_results[-1]['rerank_score']:.4f})"
    )
    return top_results


def get_top_confidence(reranked_docs: list[dict]) -> float:
    """Return the highest rerank score, used for HyDE trigger decision."""
    if not reranked_docs:
        return 0.0
    return max(doc.get("rerank_score", 0.0) for doc in reranked_docs)
