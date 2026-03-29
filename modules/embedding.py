"""
Embedding Generation Module
==============================
Generates dense vector embeddings using the BAAI/bge-large-en model
from the sentence-transformers library.

Key Design Decisions:
- Uses instruction prefix for retrieval tasks as recommended by BGE authors
- Normalizes embeddings for cosine similarity
- Supports batch encoding for efficiency
- Model loaded lazily to reduce startup time when not needed
- Thread-safe singleton pattern for model instance

References:
- https://huggingface.co/BAAI/bge-large-en
"""

import numpy as np
from typing import Optional

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Lazy model loading (singleton)
# ============================================
_embedding_model = None


def _get_model():
    """Lazily load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(
            f"Embedding model loaded successfully. "
            f"Dimension: {_embedding_model.get_sentence_embedding_dimension()}"
        )
    return _embedding_model


def generate_embedding(
    text: str,
    is_query: bool = False,
) -> np.ndarray:
    """
    Generate a normalized embedding for a single text.

    For retrieval tasks, the BGE model recommends prepending an instruction
    prefix to queries (not to documents).

    Args:
        text: Input text to embed.
        is_query: If True, prepend the retrieval instruction prefix.

    Returns:
        Normalized numpy array of shape (dimension,).
    """
    model = _get_model()

    if is_query:
        text = settings.EMBEDDING_INSTRUCTION + text

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.array(embedding, dtype=np.float32)


def generate_embeddings_batch(
    texts: list[str],
    is_query: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate normalized embeddings for a batch of texts.

    Args:
        texts: List of input texts.
        is_query: If True, prepend instruction prefix to all texts.
        batch_size: Encoding batch size (adjust based on available memory).

    Returns:
        Numpy array of shape (num_texts, dimension).
    """
    model = _get_model()

    if is_query:
        texts = [settings.EMBEDDING_INSTRUCTION + t for t in texts]

    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    return np.array(embeddings, dtype=np.float32)


def get_embedding_dimension() -> int:
    """
    Get the dimension of the embedding model output.

    Returns:
        Integer dimension size.
    """
    model = _get_model()
    return model.get_sentence_embedding_dimension()
