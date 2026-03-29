"""
Vector Store Module (ChromaDB)
================================
Manages the ChromaDB vector database for storing and querying
ticket embeddings with associated metadata.

Key Features:
- Persistent storage (survives restarts)
- Metadata storage alongside embeddings (issue number, title, labels, text)
- Dense similarity search
- Batch upsert for efficiency
- Collection management (create, reset, stats)

Design Notes:
- Uses PersistentClient for data durability
- Documents stored with their unified text for BM25 retrieval
- Embeddings pre-computed externally (not using ChromaDB's built-in embedding)
"""

from typing import Optional

import chromadb

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Singleton ChromaDB client
# ============================================
_chroma_client = None
_collection = None


def _get_client():
    """Get or create the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        settings.ensure_directories()
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_DIR)
        )
        logger.info(f"ChromaDB client initialized at {settings.CHROMA_DB_DIR}")
    return _chroma_client


def get_collection(
    collection_name: Optional[str] = None,
    reset: bool = False,
):
    """
    Get or create the ChromaDB collection for ticket embeddings.

    Args:
        collection_name: Name of the collection. Defaults to settings.
        reset: If True, delete and recreate the collection.

    Returns:
        ChromaDB Collection object.
    """
    global _collection

    if collection_name is None:
        collection_name = settings.CHROMA_COLLECTION_NAME

    client = _get_client()

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection may not exist
        _collection = None

    if _collection is None:
        _collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info(
            f"Collection '{collection_name}' ready "
            f"(contains {_collection.count()} documents)"
        )

    return _collection


def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    batch_size: int = 100,
):
    """
    Add documents with embeddings and metadata to ChromaDB.
    Uses batch upsert for efficiency.

    Args:
        ids: Unique document IDs (e.g., issue numbers).
        embeddings: List of embedding vectors.
        documents: List of document texts.
        metadatas: List of metadata dictionaries.
        batch_size: Number of documents per upsert batch.
    """
    collection = get_collection()

    total = len(ids)
    logger.info(f"Adding {total} documents to ChromaDB...")

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

        if (batch_end) % 200 == 0 or batch_end == total:
            logger.info(f"Upserted {batch_end}/{total} documents")

    logger.info(f"Successfully stored {total} documents in ChromaDB")


def query_similar(
    query_embedding: list[float],
    top_k: Optional[int] = None,
) -> dict:
    """
    Query ChromaDB for the most similar documents.

    Args:
        query_embedding: Query vector.
        top_k: Number of results to return. Defaults to settings.RETRIEVAL_TOP_K.

    Returns:
        Dictionary with keys: ids, documents, metadatas, distances.
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K

    collection = get_collection()

    # Ensure we don't request more than available
    available = collection.count()
    top_k = min(top_k, available) if available > 0 else top_k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def get_all_documents() -> dict:
    """
    Retrieve all documents from the collection.
    Used for building the BM25 index.

    Returns:
        Dictionary with keys: ids, documents, metadatas.
    """
    collection = get_collection()
    count = collection.count()

    if count == 0:
        logger.warning("Collection is empty")
        return {"ids": [], "documents": [], "metadatas": []}

    results = collection.get(
        include=["documents", "metadatas"],
    )

    logger.info(f"Retrieved {len(results['ids'])} documents from ChromaDB")

    return {
        "ids": results["ids"],
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


def get_collection_stats() -> dict:
    """
    Get statistics about the current collection.

    Returns:
        Dictionary with collection statistics.
    """
    collection = get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }
