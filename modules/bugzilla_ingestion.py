"""
Bugzilla Core Data Ingestion and Preprocessing
==============================================
Fetches Mozilla Bugzilla bugs for the Core product.

Primary workflow in this stage is RAW collection:
- Fetch bug list with pagination
- Fetch comments per bug
- Save raw bug+comment objects for later preprocessing

Preprocessing helpers are also included in this file for later stages.

Pipeline:
1. Fetch bug list from Bugzilla REST API with pagination
2. Fetch comments per bug
3. Clean and normalize text
4. Extract technical solution snippets from comments
5. Map component to team
6. Save dataset as JSON
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

# Support direct execution: python modules/bugzilla_ingestion.py
if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


BUGZILLA_BUG_URL = "https://bugzilla.mozilla.org/rest/bug"
DEFAULT_INCLUDE_FIELDS = ("id", "summary", "component", "assigned_to")

# Keep only technical comments likely containing actionable resolution details.
INCLUDE_KEYWORDS = (
    "fix",
    "patch",
    "resolved",
    "error",
    "caused by",
    "solution",
    "issue",
)
EXCLUDE_KEYWORDS = ("thanks", "duplicate", "assigned", "closing")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
BUGZILLA_ACTIVITY_PATTERN = re.compile(
    r"(?:created|updated)\s+by\s+.*?additional\s+details\s*:\s*",
    re.IGNORECASE | re.DOTALL,
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
BULK_STATUS_PATTERN = re.compile(
    r"\?\s*/\s*\(current\s+state:[^)]+\)",
    re.IGNORECASE,
)

NOISE_PHRASES = (
    "mid-air collision",
    "bugzilla cleanup",
    "mass verification",
    "reopening",
    "verified",
)


def _safe_get_json(
    url: str,
    params: Optional[dict] = None,
    max_retries: int = 4,
    timeout: int = 30,
) -> Optional[dict]:
    """Run a resilient GET request and return parsed JSON or None."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "2"))
                wait_seconds = max(retry_after, 1) + attempt
                logger.warning(
                    "Rate limited on %s. Waiting %ss before retry %s/%s",
                    url,
                    wait_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code >= 500:
                wait_seconds = 2 ** attempt
                logger.warning(
                    "Server error %s on %s. Retrying in %ss (%s/%s)",
                    response.status_code,
                    url,
                    wait_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            wait_seconds = 2 ** attempt
            logger.warning(
                "Request failed for %s (%s). Retrying in %ss (%s/%s)",
                url,
                exc,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)

    logger.error("Failed to fetch %s after %s retries", url, max_retries)
    return None


def fetch_core_bugs(
    target_count: int = 500,
    page_size: int = 100,
    delay_seconds: float = 0.25,
) -> list[dict]:
    """
    Fetch Core product bugs from Bugzilla using offset pagination.

    Returns up to target_count bugs containing id, summary, component, assigned_to.
    """
    bugs: list[dict] = []
    offset = 0

    while len(bugs) < target_count:
        params = {
            "product": "Core",
            "include_fields": ",".join(DEFAULT_INCLUDE_FIELDS),
            "limit": page_size,
            "offset": offset,
        }

        payload = _safe_get_json(BUGZILLA_BUG_URL, params=params)
        if payload is None:
            logger.error("Stopping bug list ingestion because API responses failed")
            break

        batch = payload.get("bugs", [])
        if not batch:
            logger.info("No more bugs returned by API at offset %s", offset)
            break

        bugs.extend(batch)
        logger.info("Fetched %s bugs so far...", len(bugs))

        if len(batch) < page_size:
            break

        offset += page_size
        time.sleep(delay_seconds)

    bugs = bugs[:target_count]

    if len(bugs) < target_count:
        logger.warning(
            "Fetched %s bugs, below target_count=%s",
            len(bugs),
            target_count,
        )

    return bugs


def fetch_bug_comments(bug_id: int, delay_seconds: float = 0.1) -> list[str]:
    """Fetch all comment text for a single bug ID."""
    url = f"{BUGZILLA_BUG_URL}/{bug_id}/comment"
    payload = _safe_get_json(url)
    if payload is None:
        return []

    comments_raw = payload.get("bugs", {}).get(str(bug_id), {}).get("comments", [])
    comments = [c.get("text", "").strip() for c in comments_raw if c.get("text")]

    time.sleep(delay_seconds)
    return comments


def attach_comments_to_bugs(
    bugs: list[dict],
    delay_seconds: float = 0.1,
) -> list[dict]:
    """Attach comment text lists to each bug dict under key 'comments'."""
    enriched: list[dict] = []

    for index, bug in enumerate(bugs, start=1):
        bug_id = bug.get("id")
        if bug_id is None:
            continue

        comments = fetch_bug_comments(int(bug_id), delay_seconds=delay_seconds)

        bug_copy = dict(bug)
        bug_copy["comments"] = comments
        enriched.append(bug_copy)

        if index % 50 == 0 or index == len(bugs):
            logger.info("Fetched comments for %s/%s bugs...", index, len(bugs))

    return enriched


def build_bugzilla_core_raw_data(
    target_count: int = 800,
    page_size: int = 100,
    bug_list_delay: float = 0.25,
    comment_delay: float = 0.1,
) -> list[dict]:
    """Fetch raw Bugzilla Core bugs and attach raw comments."""
    logger.info("Starting Bugzilla Core RAW ingestion (target=%s)", target_count)

    bugs = fetch_core_bugs(
        target_count=target_count,
        page_size=page_size,
        delay_seconds=bug_list_delay,
    )

    if not bugs:
        logger.warning("No bugs were fetched from Bugzilla")
        return []

    raw_bugs = attach_comments_to_bugs(bugs, delay_seconds=comment_delay)
    logger.info("Built raw Bugzilla dataset with %s records", len(raw_bugs))
    return raw_bugs


def clean_text(text: str) -> str:
    """Normalize text by removing markup/noise and lowercasing."""
    if not text:
        return ""

    text = html.unescape(text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip().lower()


def clean_comment_text(comment: str) -> str:
    """Clean Bugzilla comment text and remove common metadata wrappers."""
    if not comment:
        return ""

    text = html.unescape(comment)
    text = BUGZILLA_ACTIVITY_PATTERN.sub("", text)
    text = clean_text(text)
    text = BULK_STATUS_PATTERN.sub(" ", text)

    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, " ")

    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_description_from_comments(comments: list[str]) -> str:
    """
    Extract a clean issue description from comments.

    The first comment typically contains the original report and
    metadata wrapper; this function strips wrappers and keeps the
    first meaningful text chunk.
    """
    for comment in comments:
        cleaned = clean_comment_text(comment)
        if len(cleaned) >= 30:
            return cleaned
    return ""


def map_team_from_component(component: str) -> str:
    """
    Map Bugzilla component to team.
    Example: 'Core::Networking' -> 'Networking'.
    """
    component = (component or "").strip()
    if not component:
        return "Unknown"

    if "::" in component:
        team = component.split("::", 1)[1].strip()
        return team or "Unknown"

    # Most Bugzilla Core components use 'Domain: Subdomain' format.
    if ":" in component:
        team = component.split(":", 1)[1].strip()
        return team or component

    return component


def extract_solution_from_comments(comments: list[str]) -> str:
    """Return merged technical comments filtered by include/exclude keyword rules."""
    selected: list[str] = []
    seen: set[str] = set()

    for comment in comments:
        normalized = clean_comment_text(comment)
        if not normalized:
            continue

        if len(normalized) < 20:
            continue

        if "current state" in normalized and "no resolution" in normalized:
            continue

        if any(keyword in normalized for keyword in EXCLUDE_KEYWORDS):
            continue

        if not any(keyword in normalized for keyword in INCLUDE_KEYWORDS):
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append(normalized)

    return " ".join(selected)


def preprocess_bug_entry(bug: dict) -> dict:
    """Convert a raw bug+comments object into the final dataset shape."""
    comments = bug.get("comments", [])
    summary = bug.get("summary", "")
    title = clean_text(summary)

    description = extract_description_from_comments(comments)

    component = bug.get("component", "") or "Unknown"
    team = map_team_from_component(component)

    # First comment usually acts as the bug description; use remaining comments
    # primarily for extraction of possible fix/solution details.
    solution_comments = comments[1:] if len(comments) > 1 else comments
    solution = extract_solution_from_comments(solution_comments)

    assignee = bug.get("assigned_to") or "unassigned"

    text_parts = [
        f"title: {title}",
        f"component: {component}",
        f"team: {team}",
        f"description: {description}",
    ]
    if solution:
        text_parts.append(f"possible_fix: {solution}")

    rag_text = "\n".join(text_parts)

    return {
        "id": str(bug.get("id", "")),
        "title": title,
        "description": description,
        "text": rag_text,
        "team": team,
        "component": component,
        "assignee": assignee,
        "solution": solution,
        "num_comments": len(comments),
    }


def load_bugzilla_raw_data(filepath: Optional[str] = None) -> list[dict]:
    """Load raw Bugzilla data from JSON file."""
    if filepath is None:
        path = settings.RAW_DATA_DIR / "bugzilla_core_raw_issues.json"
    else:
        path = Path(filepath)

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    logger.info("Loaded %s raw Bugzilla records from %s", len(data), path)
    return data


def build_bugzilla_core_clean_dataset(
    raw_data_path: Optional[str] = None,
    max_records: Optional[int] = None,
) -> list[dict]:
    """Build a clean RAG-ready dataset from raw Bugzilla records.

    Records with empty solution are intentionally excluded to keep
    only actionable entries for downstream RAG retrieval.
    """
    raw_data = load_bugzilla_raw_data(raw_data_path)

    if max_records is not None:
        raw_data = raw_data[:max_records]

    cleaned: list[dict] = []
    for index, bug in enumerate(raw_data, start=1):
        if not isinstance(bug, dict):
            continue
        if bug.get("id") is None or not bug.get("summary"):
            continue

        entry = preprocess_bug_entry(bug)
        solution_text = str(entry.get("solution", "")).strip()
        if not solution_text:
            # Keep only entries with actionable resolution context.
            continue

        cleaned.append(entry)

        if index % 50 == 0 or index == len(raw_data):
            logger.info("Preprocessed %s/%s raw bugs...", index, len(raw_data))

    logger.info("Built clean Bugzilla dataset with %s records", len(cleaned))
    return cleaned


def build_bugzilla_core_dataset(
    target_count: int = 500,
    page_size: int = 100,
    bug_list_delay: float = 0.25,
    comment_delay: float = 0.1,
) -> list[dict]:
    """Fetch, enrich, and preprocess Bugzilla Core bugs into final dataset records."""
    logger.info("Starting Bugzilla Core ingestion pipeline (target=%s)", target_count)

    bugs = fetch_core_bugs(
        target_count=target_count,
        page_size=page_size,
        delay_seconds=bug_list_delay,
    )

    if not bugs:
        logger.warning("No bugs were fetched from Bugzilla")
        return []

    bugs_with_comments = attach_comments_to_bugs(bugs, delay_seconds=comment_delay)

    dataset = [preprocess_bug_entry(bug) for bug in bugs_with_comments if bug.get("id") is not None]
    dataset = [entry for entry in dataset if entry["text"]]

    logger.info("Built processed Bugzilla dataset with %s records", len(dataset))
    return dataset


def save_bugzilla_raw_data(
    raw_data: list[dict],
    output_path: Optional[str] = None,
) -> Path:
    """Save raw bug+comments data to JSON in data/raw by default."""
    if output_path is None:
        output = settings.RAW_DATA_DIR / "bugzilla_core_raw_issues.json"
    else:
        output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(raw_data, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %s raw Bugzilla records to %s", len(raw_data), output)
    return output


def save_bugzilla_dataset(
    dataset: list[dict],
    output_path: Optional[str] = None,
) -> Path:
    """Save processed dataset to JSON with UTF-8 encoding and indentation."""
    if output_path is None:
        output = settings.DATA_DIR / "bugzilla_core_dataset.json"
    else:
        output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %s Bugzilla records to %s", len(dataset), output)
    return output


def save_bugzilla_clean_dataset(
    dataset: list[dict],
    output_path: Optional[str] = None,
) -> Path:
    """Save clean Bugzilla dataset to data/processed by default."""
    if output_path is None:
        output = settings.PROCESSED_DATA_DIR / "bugzilla_core_clean_dataset.json"
    else:
        output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %s clean Bugzilla records to %s", len(dataset), output)
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Fetch raw Mozilla Bugzilla Core bugs",
    )
    parser.add_argument("--target-bugs", type=int, default=800, help="Target number of bugs")
    parser.add_argument("--page-size", type=int, default=100, help="Pagination page size")
    parser.add_argument(
        "--list-delay",
        type=float,
        default=0.25,
        help="Delay between bug list API calls in seconds",
    )
    parser.add_argument(
        "--comment-delay",
        type=float,
        default=0.1,
        help="Delay between comment API calls in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Raw output JSON path",
    )
    return parser


def main() -> None:
    """Standalone entrypoint for direct module execution."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    raw_data = build_bugzilla_core_raw_data(
        target_count=args.target_bugs,
        page_size=args.page_size,
        bug_list_delay=args.list_delay,
        comment_delay=args.comment_delay,
    )

    if not raw_data:
        logger.error("No raw records generated from Bugzilla")
        raise SystemExit(1)

    save_bugzilla_raw_data(raw_data, output_path=args.output)


if __name__ == "__main__":
    main()
