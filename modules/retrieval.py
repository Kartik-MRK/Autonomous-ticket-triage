"""
Hybrid Retrieval Module
=========================
Combines dense retrieval (ChromaDB) + sparse retrieval (BM25)
merged via Reciprocal Rank Fusion (RRF).

Public functions exposed:
- hybrid_retrieve()   — full hybrid pipeline
- dense_retrieve()    — dense-only (for tests)
- sparse_retrieve()   — sparse-only (for tests)
- rebuild_bm25_index()
"""

from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi

from modules.embedding import generate_embedding
from modules.vector_store import query_similar, get_all_documents
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_bm25_index = None
_bm25_corpus_ids = None
_bm25_corpus_docs = None
_bm25_corpus_metadatas = None


def _build_bm25_index(force_rebuild: bool = False):
    """Build the BM25 index from all documents in ChromaDB."""
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

    tokenized_corpus = [doc.lower().split() for doc in _bm25_corpus_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")


def _dense_retrieval(query_embedding: list[float], top_k: int) -> list[dict]:
    """Perform dense retrieval using ChromaDB cosine similarity."""
    results = query_similar(query_embedding, top_k=top_k)
    dense_results = []
    for i in range(len(results["ids"])):
        distance = results["distances"][i]
        similarity = 1 - distance
        dense_results.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
            "score": similarity,
            "source": "dense",
        })
    return dense_results


def _sparse_retrieval(query: str, top_k: int) -> list[dict]:
    """Perform sparse retrieval using BM25."""
    _build_bm25_index()
    if _bm25_index is None:
        logger.warning("BM25 index not available")
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    sparse_results = []
    for idx in top_indices:
        if scores[idx] > 0:
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
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    fused_scores = {}
    doc_map = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
        doc_map[doc_id] = result

    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
        if doc_id not in doc_map:
            doc_map[doc_id] = result

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        result = doc_map[doc_id].copy()
        result["rrf_score"] = fused_scores[doc_id]
        result["source"] = "hybrid"
        fused_results.append(result)
    return fused_results


# ============================================
# Public API
# ============================================

def dense_retrieve(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Dense-only retrieval. Exposed for test evaluation scripts."""
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    query_embedding = generate_embedding(query, is_query=True)
    return _dense_retrieval(query_embedding.tolist(), top_k=top_k)


def sparse_retrieve(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Sparse-only retrieval (BM25). Exposed for test evaluation scripts."""
    if top_k is None:
        top_k = settings.BM25_TOP_K
    return _sparse_retrieval(query, top_k=top_k)


def hybrid_retrieve(
    query: str,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    final_top_k: Optional[int] = None,
) -> list[dict]:
    """Perform hybrid retrieval: dense + BM25 + RRF fusion."""
    if dense_top_k is None:
        dense_top_k = settings.RETRIEVAL_TOP_K
    if sparse_top_k is None:
        sparse_top_k = settings.BM25_TOP_K
    if final_top_k is None:
        final_top_k = settings.RETRIEVAL_TOP_K

    logger.info(f"Hybrid retrieval for query: '{query[:80]}...'")

    query_embedding = generate_embedding(query, is_query=True)
    dense_results = _dense_retrieval(query_embedding.tolist(), top_k=dense_top_k)
    logger.info(f"Dense retrieval returned {len(dense_results)} results")

    sparse_results = _sparse_retrieval(query, top_k=sparse_top_k)
    logger.info(f"Sparse retrieval returned {len(sparse_results)} results")

    fused_results = _reciprocal_rank_fusion(dense_results, sparse_results, k=settings.RRF_K)
    final_results = fused_results[:final_top_k]

    logger.info(
        f"Hybrid retrieval complete: {len(final_results)} results "
        f"(from {len(dense_results)} dense + {len(sparse_results)} sparse)"
    )
    return final_results


def rebuild_bm25_index():
    """Force rebuild the BM25 index. Call after adding new documents."""
    _build_bm25_index(force_rebuild=True)
