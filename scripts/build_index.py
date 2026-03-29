"""
Build Index Script
====================
Standalone script to process raw issues and build the vector store index.

Pipeline:
1. Load raw issues from data/raw/issues.json
2. Preprocess each issue (regex + spaCy)
3. Generate embeddings (BAAI/bge-large-en)
4. Store in ChromaDB with metadata

Usage:
    python scripts/build_index.py [--reset]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ingestion import load_raw_issues
from modules.preprocessing import preprocess_batch, load_processed_issues
from modules.embedding import generate_embeddings_batch
from modules.vector_store import add_documents, get_collection, get_collection_stats
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build the vector store index from fetched issues"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before building",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing (use existing processed data)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Vector Store Index")
    logger.info("=" * 60)

    # ---- Step 1: Load or preprocess data ----
    if args.skip_preprocessing and settings.PROCESSED_ISSUES_FILE.exists():
        logger.info("Loading existing processed data...")
        processed_issues = load_processed_issues()
    else:
        logger.info("Loading raw issues...")
        raw_issues = load_raw_issues()
        logger.info(f"Loaded {len(raw_issues)} raw issues")

        logger.info("Preprocessing issues...")
        processed_issues = preprocess_batch(raw_issues)

    logger.info(f"Working with {len(processed_issues)} processed issues")

    # ---- Step 2: Reset collection if requested ----
    if args.reset:
        logger.info("Resetting vector store...")
        get_collection(reset=True)

    # ---- Step 3: Generate embeddings ----
    logger.info("Generating embeddings...")
    texts = [issue["unified_text"] for issue in processed_issues]
    embeddings = generate_embeddings_batch(texts, is_query=False, batch_size=32)

    # ---- Step 4: Prepare metadata and store ----
    ids = [str(issue["number"]) for issue in processed_issues]
    metadatas = []
    for issue in processed_issues:
        metadatas.append({
            "issue_number": str(issue["number"]),
            "title": issue.get("original_title", ""),
            "labels": ", ".join(issue.get("labels", [])),
            "state": issue.get("state", ""),
            "assignees": ", ".join(issue.get("assignees", [])),
        })

    # Convert embeddings to list format for ChromaDB
    embeddings_list = embeddings.tolist()

    logger.info("Storing in ChromaDB...")
    add_documents(
        ids=ids,
        embeddings=embeddings_list,
        documents=texts,
        metadatas=metadatas,
    )

    # ---- Step 5: Verify ----
    stats = get_collection_stats()
    logger.info(f"Vector store stats: {stats}")
    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
