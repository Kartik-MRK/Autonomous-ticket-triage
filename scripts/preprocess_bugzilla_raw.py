"""
Preprocess Raw Bugzilla Core Dataset
====================================
Reads raw Bugzilla data from data/raw and creates a clean RAG-ready dataset
in data/processed.

Usage:
    python scripts/preprocess_bugzilla_raw.py
    python scripts/preprocess_bugzilla_raw.py --max-records 30
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.bugzilla_ingestion import (  # noqa: E402
    build_bugzilla_core_clean_dataset,
    save_bugzilla_clean_dataset,
)
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw Bugzilla Core data into a clean RAG dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Raw input JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/bugzilla_core_clean_dataset.json",
        help="Processed output JSON path",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for preprocessing records",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Bugzilla raw preprocessing started")
    logger.info("=" * 60)

    cleaned = build_bugzilla_core_clean_dataset(
        raw_data_path=args.input,
        max_records=args.max_records,
    )

    if not cleaned:
        logger.error("No clean records generated")
        sys.exit(1)

    output = save_bugzilla_clean_dataset(cleaned, output_path=args.output)
    logger.info("Done. Clean records: %s", len(cleaned))
    logger.info("Saved file: %s", output)


if __name__ == "__main__":
    main()
