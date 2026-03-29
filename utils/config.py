"""
Configuration Management Module
================================
Loads environment variables from .env file and provides a centralized
Settings class for all configurable parameters across the application.

Supports easy switching between models, adjusting retrieval parameters,
and scaling data ingestion without code changes.
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

    # ---- API Keys ----
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # ---- Model Configuration ----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # ---- Retrieval Configuration ----
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
    RRF_K: int = 60  # Reciprocal Rank Fusion constant

    # ---- GitHub Data Configuration ----
    GITHUB_REPO_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "microsoft")
    GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "vscode")
    MAX_ISSUES: int = int(os.getenv("MAX_ISSUES", "500"))
    GITHUB_GRAPHQL_URL: str = "https://api.github.com/graphql"

    # ---- Data Paths ----
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

    RAW_ISSUES_FILE: Path = RAW_DATA_DIR / "issues.json"
    PROCESSED_ISSUES_FILE: Path = PROCESSED_DATA_DIR / "issues_processed.json"

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
        if not cls.GITHUB_TOKEN:
            warnings.append("GITHUB_TOKEN is not set - data ingestion will fail")
        if not cls.GOOGLE_API_KEY:
            warnings.append("GOOGLE_API_KEY is not set - classification/generation will fail")
        return warnings


# Create a singleton instance
settings = Settings()
