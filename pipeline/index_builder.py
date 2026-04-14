"""
Index Builder
==============
Builds the ChromaDB index from Bugzilla data using the
regex + spaCy preprocessing path. Also creates the test split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from modules.ingestion import (
    build_bugzilla_core_clean_dataset,
    save_bugzilla_clean_dataset,
)
from modules.preprocessing import preprocess_batch
from modules.embedding import generate_embeddings_batch
from modules.vector_store import add_documents, get_collection, get_collection_stats
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _to_preprocess_issue(record: dict, fallback_number: int) -> dict:
    """Convert a Bugzilla clean record into the common issue preprocessing schema."""
    bug_id = str(record.get("id", "")).strip()
    number = int(bug_id) if bug_id.isdigit() else fallback_number

    title = str(record.get("title", "")).strip()
    description = str(record.get("description", "")).strip()
    solution = str(record.get("solution", "")).strip()
    full_text = str(record.get("text", "")).strip()

    body_parts = [description, solution, full_text]
    body = "\n\n".join(part for part in body_parts if part)

    component = str(record.get("component", "")).strip()
    team = str(record.get("team", "")).strip()
    labels = []
    if component:
        labels.append(f"component:{component}")
    if team:
        labels.append(f"team:{team}")

    assignee = str(record.get("assignee", "")).strip()
    comments = [{"body": solution}] if solution else []

    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": labels,
        "comments": comments,
        "assignees": [assignee] if assignee else [],
        "state": "resolved" if solution else "unknown",
        "created_at": "",
        "updated_at": "",
    }


def _split_test_set(
    clean_data: list[dict],
    test_count: int = 100,
) -> tuple[list[dict], list[dict]]:
    """
    Randomly sample `test_count` issues for test_processed.json.
    Returns (train_data, test_data).
    """
    if len(clean_data) <= test_count:
        logger.warning(
            f"Dataset has only {len(clean_data)} records, "
            f"cannot split {test_count} for testing. Using all for index."
        )
        return clean_data, []

    random.seed(42)
    indices = list(range(len(clean_data)))
    random.shuffle(indices)
    test_indices = set(indices[:test_count])

    train_data = [clean_data[i] for i in range(len(clean_data)) if i not in test_indices]
    test_data = [clean_data[i] for i in sorted(test_indices)]

    logger.info(f"Split: {len(train_data)} train + {len(test_data)} test records")
    return train_data, test_data


def build_index(
    raw_input: str = None,
    clean_input: str = None,
    processed_output: str = None,
    max_records: int | None = None,
    rebuild_clean: bool = False,
    reset: bool = False,
    test_split_count: int | None = None,
) -> dict:
    """
    Build ChromaDB index from Bugzilla data.

    Also creates test_processed.json with `test_split_count` held-out issues.
    Returns summary dictionary with counts and collection stats.
    """
    if raw_input is None:
        raw_input = str(settings.RAW_ISSUES_FILE)
    if clean_input is None:
        clean_input = str(settings.CLEAN_DATASET_FILE)
    if processed_output is None:
        processed_output = str(settings.PROCESSED_ISSUES_FILE)
    if test_split_count is None:
        test_split_count = settings.TEST_SPLIT_COUNT

    logger.info("=" * 60)
    logger.info("Building Bugzilla Chroma index (regex+spaCy path)")
    logger.info("=" * 60)

    clean_path = Path(clean_input)

    # Step 1: Build or load clean dataset
    if rebuild_clean or not clean_path.exists():
        logger.info("Building clean Bugzilla dataset from raw input: %s", raw_input)
        bugzilla_clean = build_bugzilla_core_clean_dataset(
            raw_data_path=raw_input,
            max_records=max_records,
        )
        if not bugzilla_clean:
            raise RuntimeError("Failed to build clean Bugzilla dataset from raw input")
        save_bugzilla_clean_dataset(bugzilla_clean, output_path=str(clean_path))
        logger.info("Saved clean Bugzilla dataset to %s", clean_path)
    else:
        logger.info("Loading existing clean Bugzilla dataset: %s", clean_path)
        with open(clean_path, "r", encoding="utf-8") as handle:
            bugzilla_clean = json.load(handle)

    if not isinstance(bugzilla_clean, list) or not bugzilla_clean:
        raise RuntimeError(f"Input dataset is empty or invalid: {clean_path}")

    logger.info("Loaded %s clean Bugzilla records", len(bugzilla_clean))

    # Step 2: Split test set
    train_data, test_data = _split_test_set(bugzilla_clean, test_count=test_split_count)

    if test_data:
        test_output = settings.TEST_PROCESSED_FILE
        test_output.parent.mkdir(parents=True, exist_ok=True)
        with open(test_output, "w", encoding="utf-8") as handle:
            json.dump(test_data, handle, indent=2, ensure_ascii=False)
        logger.info("Saved %s test records to %s", len(test_data), test_output)

    # Step 3: Adapt train records for preprocessing
    adapted_issues = []
    source_by_number: dict[str, dict] = {}

    for idx, record in enumerate(train_data, start=1):
        if not isinstance(record, dict):
            continue
        issue = _to_preprocess_issue(record, fallback_number=idx)
        adapted_issues.append(issue)
        source_by_number[str(issue["number"])] = {
            "component": str(record.get("component", "")),
            "team": str(record.get("team", "")),
            "assignee": str(record.get("assignee", "")),
            "num_comments": int(record.get("num_comments", 0) or 0),
        }

    if not adapted_issues:
        raise RuntimeError("No valid Bugzilla records were adapted for preprocessing")

    # Step 4: regex+spaCy preprocessing
    logger.info("Running regex+spaCy preprocessing on %s records...", len(adapted_issues))
    processed_issues = preprocess_batch(
        adapted_issues,
        save_output=True,
        output_path=processed_output,
    )

    if not processed_issues:
        raise RuntimeError("No records left after regex+spaCy preprocessing")

    # Step 5: Reset Chroma if requested
    if reset:
        logger.info("Resetting Chroma collection before upsert...")
        get_collection(reset=True)

    # Step 6: Generate embeddings
    logger.info("Generating embeddings for processed Bugzilla records...")
    texts = [issue["unified_text"] for issue in processed_issues]
    embeddings = generate_embeddings_batch(texts, is_query=False, batch_size=32)

    # Step 7: Upsert into ChromaDB
    ids = [f"bugzilla-{issue['number']}" for issue in processed_issues]
    metadatas = []
    for issue in processed_issues:
        key = str(issue["number"])
        src = source_by_number.get(key, {})
        metadatas.append({
            "issue_number": key,
            "source": "bugzilla",
            "title": issue.get("original_title", ""),
            "labels": ", ".join(issue.get("labels", [])),
            "state": issue.get("state", ""),
            "assignees": ", ".join(issue.get("assignees", [])),
            "component": str(src.get("component", "")),
            "team": str(src.get("team", "")),
            "assignee": str(src.get("assignee", "")),
            "num_comments": int(src.get("num_comments", 0)),
        })

    logger.info("Upserting %s records into ChromaDB...", len(ids))
    add_documents(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    stats = get_collection_stats()
    logger.info("Vector store stats: %s", stats)
    logger.info("=" * 60)
    logger.info("Index build complete")
    logger.info("=" * 60)

    return {
        "total_clean": len(bugzilla_clean),
        "train_count": len(train_data),
        "test_count": len(test_data),
        "spacy_processed_count": len(processed_issues),
        "collection_stats": stats,
        "clean_path": str(clean_path),
        "processed_output": processed_output,
        "test_output": str(settings.TEST_PROCESSED_FILE) if test_data else None,
    }
