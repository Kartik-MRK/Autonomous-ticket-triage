"""
Ingest Bugzilla Core Dataset
============================
Fetches raw bugs from Mozilla Bugzilla REST API and writes raw JSON output
into data/raw for later preprocessing.

Usage:
    python scripts/ingest_bugzilla_core.py --target-bugs 200
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.bugzilla_ingestion import (  # noqa: E402
    build_bugzilla_core_raw_data,
    save_bugzilla_raw_data,
)
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch raw Bugzilla Core bugs for later preprocessing"
    )
    parser.add_argument(
        "--target-bugs",
        type=int,
        default=800,
        help="Target number of bugs to fetch (default: 800)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Bug list page size for offset pagination (default: 100)",
    )
    parser.add_argument(
        "--list-delay",
        type=float,
        default=0.25,
        help="Delay in seconds between bug list API calls (default: 0.25)",
    )
    parser.add_argument(
        "--comment-delay",
        type=float,
        default=0.1,
        help="Delay in seconds between comment API calls (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Output JSON path (default: data/raw/bugzilla_core_raw_issues.json)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Bugzilla Core ingestion started")
    logger.info("=" * 60)

    raw_data = build_bugzilla_core_raw_data(
        target_count=args.target_bugs,
        page_size=args.page_size,
        bug_list_delay=args.list_delay,
        comment_delay=args.comment_delay,
    )

    if not raw_data:
        logger.error("No raw records produced")
        sys.exit(1)

    output = save_bugzilla_raw_data(raw_data, output_path=args.output)
    logger.info("Done. Raw records: %s", len(raw_data))
    logger.info("Saved file: %s", output)


if __name__ == "__main__":
    main()
