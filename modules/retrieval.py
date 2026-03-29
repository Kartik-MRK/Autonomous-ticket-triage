"""
Hybrid Retrieval Module
=========================
Implements a hybrid retrieval system combining:
1. Dense retrieval: Cosine similarity search via ChromaDB embeddings
2. Sparse retrieval: BM25 keyword matching via rank-bm25

Results are merged using Reciprocal Rank Fusion (RRF) to produce
a unified ranked list that leverages both semantic and lexical signals.

Key Design Decisions:
- RRF constant k=60 (standard value from the original paper)
- BM25 index built lazily from ChromaDB documents
- Deduplication by document ID
- Configurable top-K for both retrieval methods

References:
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from modules.embedding import generate_embedding
from modules.vector_store import query_similar, get_all_documents
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# BM25 Index (built lazily)
# ============================================
_bm25_index = None
_bm25_corpus_ids = None
_bm25_corpus_docs = None
_bm25_corpus_metadatas = None


def _build_bm25_index(force_rebuild: bool = False):
    """
    Build the BM25 index from all documents in ChromaDB.
    Index is cached for reuse across queries.

    Args:
        force_rebuild: If True, rebuild even if index already exists.
    """
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metadatas

    if _bm25_index is not None and not force_rebuild:
        return

    logger.info("Building BM25 index from ChromaDB documents...")

    all_docs = get_all_documents()

    if not all_docs["ids"]:
        logger.warning("No documents found in ChromaDB for BM25 index")
        return

    _bm25_corpus_ids = all_docs["ids"]
    _bm25_corpus_docs = all_docs["documents"]
    _bm25_corpus_metadatas = all_docs["metadatas"]

    # Tokenize documents for BM25 (simple whitespace tokenization)
    tokenized_corpus = [doc.lower().split() for doc in _bm25_corpus_docs]

    _bm25_index = BM25Okapi(tokenized_corpus)

    logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")


def _dense_retrieval(
    query_embedding: list[float],
    top_k: int,
) -> list[dict]:
    """
    Perform dense retrieval using ChromaDB cosine similarity.

    Args:
        query_embedding: Query vector.
        top_k: Number of results.

    Returns:
        List of result dictionaries with id, document, metadata, score.
    """
    results = query_similar(query_embedding, top_k=top_k)

    dense_results = []
    for i in range(len(results["ids"])):
        # ChromaDB returns distances (lower is better for cosine)
        # Convert to similarity score (higher is better)
        distance = results["distances"][i]
        similarity = 1 - distance  # cosine distance to similarity

        dense_results.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
            "score": similarity,
            "source": "dense",
        })

    return dense_results


def _sparse_retrieval(
    query: str,
    top_k: int,
) -> list[dict]:
    """
    Perform sparse retrieval using BM25.

    Args:
        query: Query text string.
        top_k: Number of results.

    Returns:
        List of result dictionaries with id, document, metadata, score.
    """
    _build_bm25_index()

    if _bm25_index is None:
        logger.warning("BM25 index not available")
        return []

    # Tokenize query
    tokenized_query = query.lower().split()

    # Get BM25 scores for all documents
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    sparse_results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include non-zero scores
            sparse_results.append({
                "id": _bm25_corpus_ids[idx],
                "document": _bm25_corpus_docs[idx],
                "metadata": _bm25_corpus_metadatas[idx],
                "score": float(scores[idx]),
                "source": "sparse",
            })

    return sparse_results


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1/(k + rank) across all result lists.

    Args:
        dense_results: Results from dense retrieval.
        sparse_results: Results from sparse retrieval.
        k: RRF constant (default 60, from original paper).

    Returns:
        Merged and sorted list of results.
    """
    # Build score map keyed by document ID
    fused_scores = {}
    doc_map = {}

    # Process dense results
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
        doc_map[doc_id] = result  # Store full result info

    # Process sparse results
    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
        if doc_id not in doc_map:
            doc_map[doc_id] = result

    # Sort by fused score (descending)
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    # Build fused results list
    fused_results = []
    for doc_id in sorted_ids:
        result = doc_map[doc_id].copy()
        result["rrf_score"] = fused_scores[doc_id]
        result["source"] = "hybrid"
        fused_results.append(result)

    return fused_results


def hybrid_retrieve(
    query: str,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    final_top_k: Optional[int] = None,
) -> list[dict]:
    """
    Perform hybrid retrieval combining dense and sparse methods with RRF fusion.

    Pipeline:
    1. Generate query embedding
    2. Dense retrieval (ChromaDB cosine similarity)
    3. Sparse retrieval (BM25 keyword matching)
    4. Reciprocal Rank Fusion to merge and rank
    5. Return top-K fused results

    Args:
        query: Query text string.
        dense_top_k: Number of dense results. Defaults to settings.RETRIEVAL_TOP_K.
        sparse_top_k: Number of sparse results. Defaults to settings.BM25_TOP_K.
        final_top_k: Number of final fused results. Defaults to settings.RETRIEVAL_TOP_K.

    Returns:
        List of ranked result dictionaries with rrf_score.
    """
    if dense_top_k is None:
        dense_top_k = settings.RETRIEVAL_TOP_K
    if sparse_top_k is None:
        sparse_top_k = settings.BM25_TOP_K
    if final_top_k is None:
        final_top_k = settings.RETRIEVAL_TOP_K

    logger.info(f"Hybrid retrieval for query: '{query[:80]}...'")

    # Step 1: Generate query embedding
    query_embedding = generate_embedding(query, is_query=True)

    # Step 2: Dense retrieval
    dense_results = _dense_retrieval(query_embedding.tolist(), top_k=dense_top_k)
    logger.info(f"Dense retrieval returned {len(dense_results)} results")

    # Step 3: Sparse retrieval (BM25)
    sparse_results = _sparse_retrieval(query, top_k=sparse_top_k)
    logger.info(f"Sparse retrieval returned {len(sparse_results)} results")

    # Step 4: Reciprocal Rank Fusion
    fused_results = _reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=settings.RRF_K,
    )

    # Step 5: Trim to final top-K
    final_results = fused_results[:final_top_k]

    logger.info(
        f"Hybrid retrieval complete: {len(final_results)} results "
        f"(from {len(dense_results)} dense + {len(sparse_results)} sparse)"
    )

    return final_results


def rebuild_bm25_index():
    """Force rebuild the BM25 index. Call after adding new documents."""
    _build_bm25_index(force_rebuild=True)
