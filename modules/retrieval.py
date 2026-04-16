"""
Hybrid Retrieval Module
=========================
Combines dense retrieval (ChromaDB) + sparse retrieval (BM25)
merged via Weighted Reciprocal Rank Fusion (WRRF).

Improvements applied:
  - Improvement 2: Weighted RRF — dense results get a configurable
    weight multiplier (default 1.2×) since BAAI/bge-large-en carries
    stronger semantic signal than raw BM25 token overlap.
  - Improvement 3: Title-boosted BM25 — document titles are prepended
    3× in the BM25 index so their TF weight reflects their relevance
    signal strength (titles are the most query-like text).
  - Improvement 4: spaCy query expansion — BM25 queries use lemmatized
    tokens from spaCy (same preprocessing as corpus) instead of raw
    whitespace splits, aligning tokenization between query and corpus.
  - Improvement 5: Larger candidate pools — dense=30, sparse=30 for
    richer RRF overlap signal.

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
    """
    Build the BM25 index from all documents in ChromaDB.

    Improvement 3 — Title Boosting:
    Each document's title is prepended twice to the indexable text so that
    its term frequency (TF) contributes 3× as much as in the raw body.
    Titles are short, precise, and the most query-like text in the corpus.
    """
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metadatas

    if _bm25_index is not None and not force_rebuild:
        return

    logger.info("Building BM25 index from ChromaDB documents (with title boosting)...")
    all_docs = get_all_documents()
    if not all_docs["ids"]:
        logger.warning("No documents found in ChromaDB for BM25 index")
        return

    _bm25_corpus_ids = all_docs["ids"]
    _bm25_corpus_docs = all_docs["documents"]
    _bm25_corpus_metadatas = all_docs["metadatas"]

    # --- Improvement 3: Title-boosted tokenization ---
    # Prepending the title twice makes it appear 3× in the final token stream.
    # This boosts TF for title terms without modifying stored documents.
    boosted_texts = []
    for doc, meta in zip(_bm25_corpus_docs, _bm25_corpus_metadatas):
        title = ""
        if isinstance(meta, dict):
            title = str(meta.get("title", "")).strip()
        if title:
            # title appears at positions 1,2 (prepended) and again inside unified_text
            boosted = f"{title} {title} {doc}"
        else:
            boosted = doc
        boosted_texts.append(boosted)

    tokenized_corpus = [text.lower().split() for text in boosted_texts]
    _bm25_index = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built with {len(tokenized_corpus)} documents (title-boosted)")


def _get_spacy_tokens(query: str) -> list[str]:
    """
    Improvement 4 — SpaCy Query Expansion:
    Tokenize the query using spaCy lemmatization instead of whitespace
    splitting. This aligns the query's representation with what was
    stored in the corpus (unified_text is already preprocessed).

    Falls back to lowercased whitespace split if spaCy fails.
    """
    try:
        from modules.preprocessing import clean_text_regex, process_with_spacy
        clean_q = clean_text_regex(query)
        spacy_result = process_with_spacy(clean_q)
        tokens = spacy_result.get("tokens", [])
        if tokens:
            return tokens
    except Exception as e:
        logger.debug(f"spaCy tokenization failed, falling back to split: {e}")
    return query.lower().split()


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
    """
    Perform sparse retrieval using BM25.

    Improvement 4: Uses spaCy lemmatized tokens for query tokenization
    to match the preprocessing used during document ingestion.
    """
    _build_bm25_index()
    if _bm25_index is None:
        logger.warning("BM25 index not available")
        return []

    # --- Improvement 4: spaCy lemmatized query tokens ---
    tokenized_query = _get_spacy_tokens(query)
    logger.debug(f"BM25 query tokens ({len(tokenized_query)}): {tokenized_query[:15]}")

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
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    Merge dense and sparse results using Weighted Reciprocal Rank Fusion.

    Improvement 2 — Weighted RRF:
    Standard RRF (Cormack et al., 2009):
        score(d) = Σ  1 / (k + rank(d))

    Weighted RRF:
        score(d) = dense_weight  × Σ_dense  1/(k + rank)
                 + sparse_weight × Σ_sparse 1/(k + rank)

    Using dense_weight=1.2, sparse_weight=1.0 slightly favours the
    dense model (BAAI/bge-large-en 1024-dim) which carries stronger
    semantic signal for technical issue text.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_score = dense_weight / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
        doc_map[doc_id] = result

    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        rrf_score = sparse_weight / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
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
    """
    Perform hybrid retrieval: dense + BM25 + Weighted RRF fusion.

    Steps:
    1. Generate query embedding (BAAI/bge-large-en with instruction prefix)
    2. Dense retrieval from ChromaDB (cosine similarity, top-30)
    3. Sparse retrieval using BM25 with spaCy tokens (top-30)
    4. Weighted RRF fusion (dense_weight=1.2, sparse_weight=1.0)
    5. Return top-N fused candidates for the reranker
    """
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

    fused_results = _reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=settings.RRF_K,
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
    )
    final_results = fused_results[:final_top_k]

    logger.info(
        f"Hybrid retrieval complete: {len(final_results)} results "
        f"(from {len(dense_results)} dense + {len(sparse_results)} sparse, "
        f"weights: dense={settings.DENSE_WEIGHT}, sparse={settings.SPARSE_WEIGHT})"
    )
    return final_results


def rebuild_bm25_index():
    """Force rebuild the BM25 index. Call after adding new documents."""
    _build_bm25_index(force_rebuild=True)
