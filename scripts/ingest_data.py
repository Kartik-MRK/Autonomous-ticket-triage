"""
Data Ingestion Script
======================
Standalone script to fetch issues from GitHub and save to disk.

Usage:
    python scripts/ingest_data.py [--max-issues 500]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ingestion import fetch_issues, save_issues
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issues for the ticket triage system"
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=settings.MAX_ISSUES,
        help=f"Maximum number of issues to fetch (default: {settings.MAX_ISSUES})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/issues.json)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GitHub Issue Ingestion Script")
    logger.info("=" * 60)
    logger.info(f"Repository: {settings.GITHUB_REPO_OWNER}/{settings.GITHUB_REPO_NAME}")
    logger.info(f"Max issues: {args.max_issues}")

    # Validate GitHub token
    if not settings.GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN is not set in .env file!")
        sys.exit(1)

    # Fetch issues
    issues = fetch_issues(max_issues=args.max_issues)

    # Save to file
    output_path = save_issues(issues, filepath=args.output)

    logger.info(f"Done! {len(issues)} issues saved to {output_path}")


if __name__ == "__main__":
    main()
