"""
Configuration Management Module
================================
Loads environment variables from .env and provides a centralized
Settings class for all configurable parameters.

All LLM calls use Ollama (llama3.1:8b) running locally.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# Load .env file from project root
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Settings:
    """
    Centralized configuration loaded from environment variables.
    All settings have sensible defaults and can be overridden via .env file.
    """

    # ---- Ollama Configuration ----
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    # ---- Model Configuration ----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # ---- Retrieval Configuration ----
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "30"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "30"))
    RRF_K: int = 60  # Reciprocal Rank Fusion smoothing constant (Cormack et al.)
    # Weighted RRF: dense model is a stronger semantic signal, give it slight boost
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "1.2"))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", "1.0"))

    # ---- HyDE Configuration ----
    HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() in ("true", "1", "yes")
    HYDE_CONFIDENCE_THRESHOLD: float = float(os.getenv("HYDE_CONFIDENCE_THRESHOLD", "0.3"))
    HYDE_AGREEMENT_THRESHOLD: float = float(os.getenv("HYDE_AGREEMENT_THRESHOLD", "0.4"))

    # ---- Bugzilla Data Configuration ----
    MAX_ISSUES: int = int(os.getenv("MAX_ISSUES", "3000"))
    TEST_SPLIT_COUNT: int = int(os.getenv("TEST_SPLIT_COUNT", "100"))

    # ---- Data Paths ----
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

    RAW_ISSUES_FILE: Path = RAW_DATA_DIR / "bugzilla_core_raw.json"
    CLEAN_DATASET_FILE: Path = PROCESSED_DATA_DIR / "bugzilla_core_clean.json"
    PROCESSED_ISSUES_FILE: Path = PROCESSED_DATA_DIR / "bugzilla_core_processed.json"
    TEST_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "test_processed.json"

    # ---- ChromaDB Configuration ----
    CHROMA_COLLECTION_NAME: str = "ticket_embeddings"

    # ---- Server Configuration ----
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ---- Logging ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ---- Embedding Configuration ----
    EMBEDDING_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "
    EMBEDDING_DIMENSION: int = 1024  # BAAI/bge-large-en output dimension

    @classmethod
    def ensure_directories(cls):
        """Create all necessary data directories if they don't exist."""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate that critical configuration is present."""
        warnings = []
        # Check Ollama connectivity
        try:
            import requests
            resp = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code != 200:
                warnings.append(f"Ollama server at {cls.OLLAMA_BASE_URL} returned status {resp.status_code}")
        except Exception:
            warnings.append(f"Cannot reach Ollama server at {cls.OLLAMA_BASE_URL}")
        return warnings


# Create singleton instance
settings = Settings()
