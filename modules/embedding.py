"""
Embedding Generation Module
==============================
Generates dense vector embeddings using BAAI/bge-large-en.
"""

import numpy as np
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model = None


def _get_model():
    """Lazily load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(
            f"Embedding model loaded. "
            f"Dimension: {_embedding_model.get_sentence_embedding_dimension()}"
        )
    return _embedding_model


def generate_embedding(text: str, is_query: bool = False) -> np.ndarray:
    """Generate a normalized embedding for a single text."""
    model = _get_model()
    if is_query:
        text = settings.EMBEDDING_INSTRUCTION + text
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embedding, dtype=np.float32)


def generate_embeddings_batch(
    texts: list[str],
    is_query: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate normalized embeddings for a batch of texts."""
    model = _get_model()
    if is_query:
        texts = [settings.EMBEDDING_INSTRUCTION + t for t in texts]
    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return np.array(embeddings, dtype=np.float32)


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding model output."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
