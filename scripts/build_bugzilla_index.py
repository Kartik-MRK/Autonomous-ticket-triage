"""
Build Bugzilla Vector Store Index
=================================
CLI wrapper to build ChromaDB index from Bugzilla data while explicitly
using regex + spaCy preprocessing.

Usage:
    python scripts/build_bugzilla_index.py --rebuild-clean --reset
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.bugzilla_index_builder import build_bugzilla_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from Bugzilla data using regex+spaCy preprocessing"
    )
    parser.add_argument(
        "--raw-input",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Path to raw Bugzilla dataset JSON",
    )
    parser.add_argument(
        "--clean-input",
        type=str,
        default="data/processed/bugzilla_core_clean_dataset.json",
        help="Path to clean Bugzilla dataset JSON",
    )
    parser.add_argument(
        "--processed-output",
        type=str,
        default="data/processed/bugzilla_core_spacy_processed.json",
        help="Where to save regex+spaCy processed Bugzilla issues",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for clean dataset build from raw data",
    )
    parser.add_argument(
        "--rebuild-clean",
        action="store_true",
        help="Force rebuilding clean dataset from raw input",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset ChromaDB collection before inserting documents",
    )

    args = parser.parse_args()

    build_bugzilla_index(
        raw_input=args.raw_input,
        clean_input=args.clean_input,
        processed_output=args.processed_output,
        max_records=args.max_records,
        rebuild_clean=args.rebuild_clean,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
