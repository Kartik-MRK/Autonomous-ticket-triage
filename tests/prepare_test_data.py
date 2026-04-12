"""Prepare Bugzilla Core raw + preprocessed test datasets under tests/data."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import requests


BUG_API_URL = "https://bugzilla.mozilla.org/rest/bug"
COMMENT_API_URL_TEMPLATE = "https://bugzilla.mozilla.org/rest/bug/{bug_id}/comment"

TARGET_VALID_SAMPLES = 150
BATCH_SIZE = 50
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.25

EXISTING_DATASET_CANDIDATES = [
    os.path.join("data", "bugzilla_core_raw_issues.json"),
    os.path.join("data", "raw", "bugzilla_core_raw_issues.json"),
]

OUTPUT_DIR = os.path.join("tests", "data")
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_issues.json")
PREPROCESSED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "preprocessed.json")

TECHNICAL_KEYWORDS = (
    "fix",
    "patch",
    "error",
    "issue",
    "caused",
    "resolve",
    "failure",
)

NON_INFORMATIVE_KEYWORDS = (
    "thanks",
    "thank you",
    "duplicate",
    "assigned",
    "closing",
    "closed",
)


def clean_text(text: str) -> str:
    """Lowercase text, remove URLs, and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_existing_issue_state() -> Tuple[Set[int], int, Optional[str], Optional[str]]:
    """Load existing IDs and the newest known creation_time threshold."""
    for path in EXISTING_DATASET_CANDIDATES:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            records = payload.get("bugs", payload.get("issues", []))
        elif isinstance(payload, list):
            records = payload
        else:
            records = []

        existing_ids: Set[int] = set()
        max_existing_id = 0
        newest_creation_time: Optional[str] = None

        for issue in records:
            if not isinstance(issue, dict):
                continue

            try:
                issue_id = int(issue.get("id"))
            except (TypeError, ValueError):
                continue

            existing_ids.add(issue_id)
            max_existing_id = max(max_existing_id, issue_id)

            creation_time = issue.get("creation_time")
            if isinstance(creation_time, str):
                if newest_creation_time is None or creation_time > newest_creation_time:
                    newest_creation_time = creation_time

        print(
            f"Loaded {len(existing_ids)} existing issue IDs from {path}. "
            f"Newest creation_time={newest_creation_time or 'n/a'}, max_id={max_existing_id}."
        )
        return existing_ids, max_existing_id, newest_creation_time, path

    print("Existing dataset not found; continuing with empty overlap set.")
    return set(), 0, None, None


def request_json(
    url: str,
    params: Optional[Dict[str, object]] = None,
    retries: int = MAX_RETRIES,
) -> Optional[Dict[str, object]]:
    """GET JSON with retry/backoff and light client-side rate limiting."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                time.sleep(REQUEST_DELAY_SECONDS)
                return response.json()

            print(f"Request failed ({response.status_code}) for {response.url}")
        except requests.RequestException as exc:
            print(f"Request error on attempt {attempt}/{retries}: {exc}")

        if attempt < retries:
            backoff_seconds = 2 ** (attempt - 1)
            print(f"Retrying in {backoff_seconds}s...")
            time.sleep(backoff_seconds)

    time.sleep(REQUEST_DELAY_SECONDS)
    return None


def fetch_bugs(offset: int, limit: int = BATCH_SIZE) -> List[Dict[str, object]]:
    """Fetch one page of recent Core bugs ordered by newest creation_time."""
    params = {
        "product": "Core",
        "include_fields": "id,summary,component,assigned_to,creation_time",
        "order": "creation_time DESC",
        "limit": limit,
        "offset": offset,
    }

    payload = request_json(BUG_API_URL, params=params)
    if not payload:
        return []

    bugs = payload.get("bugs", [])
    if not isinstance(bugs, list):
        return []

    bugs.sort(key=lambda bug: str(bug.get("creation_time", "")), reverse=True)
    return bugs


def fetch_comments(bug_id: int) -> List[str]:
    """Fetch comments for a bug and return comment text entries."""
    url = COMMENT_API_URL_TEMPLATE.format(bug_id=bug_id)
    payload = request_json(url)
    if not payload:
        return []

    bug_comment_block = payload.get("bugs", {}).get(str(bug_id), {})
    comments = bug_comment_block.get("comments", [])
    if not isinstance(comments, list):
        return []

    output: List[str] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        comment_text = comment.get("text")
        if isinstance(comment_text, str) and comment_text.strip():
            output.append(comment_text)

    return output


def extract_solution(comments: List[str]) -> str:
    """Keep technical comments and combine them into one solution field."""
    kept: List[str] = []

    for comment in comments:
        cleaned = clean_text(comment)
        if not cleaned:
            continue

        has_technical_signal = any(keyword in cleaned for keyword in TECHNICAL_KEYWORDS)
        if not has_technical_signal:
            continue

        mostly_non_informative = (
            any(keyword in cleaned for keyword in NON_INFORMATIVE_KEYWORDS)
            and len(cleaned.split()) < 16
        )
        if mostly_non_informative:
            continue

        kept.append(cleaned)

    unique_kept: List[str] = []
    seen: Set[str] = set()
    for item in kept:
        if item in seen:
            continue
        seen.add(item)
        unique_kept.append(item)

    return " ".join(unique_kept).strip()


def preprocess_text(summary: str, description: str = "") -> str:
    """Combine summary + description and normalize for RAG/eval query use."""
    combined = f"{summary or ''} {description or ''}".strip()
    return clean_text(combined)


def extract_team(component: str) -> str:
    """Infer a team-like label from component using the last :: segment."""
    if not component:
        return "unknown"
    parts = component.split("::")
    return clean_text(parts[-1]) if parts else "unknown"


def is_newer_than_existing(
    bug: Dict[str, object],
    newest_creation_time: Optional[str],
    max_existing_id: int,
) -> bool:
    """Check whether a bug is newer than known training data."""
    creation_time = bug.get("creation_time")
    bug_id = bug.get("id")

    if newest_creation_time and isinstance(creation_time, str):
        return creation_time > newest_creation_time

    try:
        bug_id_int = int(bug_id)
    except (TypeError, ValueError):
        return False

    return bug_id_int > max_existing_id


def prepare_datasets(
    target_valid_count: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fetch bugs and build both raw and filtered preprocessed test datasets."""
    existing_ids, max_existing_id, newest_creation_time, _ = load_existing_issue_state()

    raw_records: List[Dict[str, object]] = []
    preprocessed_records: List[Dict[str, object]] = []
    seen_ids: Set[int] = set(existing_ids)

    offset = 0
    total_fetched = 0
    duplicate_skipped = 0
    old_skipped = 0
    empty_solution_skipped = 0

    while len(preprocessed_records) < target_valid_count:
        bugs = fetch_bugs(offset=offset, limit=BATCH_SIZE)
        if not bugs:
            print("No more bugs returned by API; stopping.")
            break

        total_fetched += len(bugs)

        for bug in bugs:
            if len(preprocessed_records) >= target_valid_count:
                break

            try:
                bug_id = int(bug.get("id"))
            except (TypeError, ValueError):
                continue

            if bug_id in seen_ids:
                duplicate_skipped += 1
                continue

            if not is_newer_than_existing(bug, newest_creation_time, max_existing_id):
                old_skipped += 1
                continue

            comments = fetch_comments(bug_id)

            summary = str(bug.get("summary", "")).strip()
            component = str(bug.get("component", "")).strip()
            assignee = str(bug.get("assigned_to", "")).strip() or "unassigned"
            creation_time = str(bug.get("creation_time", "")).strip()

            raw_records.append(
                {
                    "id": bug_id,
                    "summary": summary,
                    "component": component,
                    "assigned_to": assignee,
                    "creation_time": creation_time,
                    "comments": comments,
                }
            )

            solution = extract_solution(comments)
            if not solution:
                empty_solution_skipped += 1
                seen_ids.add(bug_id)
                continue

            description = comments[0] if comments else ""
            text = preprocess_text(summary=summary, description=description)
            if not text:
                seen_ids.add(bug_id)
                continue

            preprocessed_records.append(
                {
                    "id": bug_id,
                    "text": text,
                    "team": extract_team(component),
                    "component": component,
                    "assignee": assignee,
                    "solution": solution,
                }
            )
            seen_ids.add(bug_id)

        print(
            "Progress: "
            f"fetched={total_fetched}, "
            f"raw_saved={len(raw_records)}, "
            f"preprocessed_valid={len(preprocessed_records)}/{target_valid_count}, "
            f"duplicates_skipped={duplicate_skipped}, "
            f"older_skipped={old_skipped}, "
            f"empty_solution_skipped={empty_solution_skipped}."
        )

        offset += BATCH_SIZE

    if len(preprocessed_records) < target_valid_count:
        raise RuntimeError(
            f"Collected only {len(preprocessed_records)} valid samples; target={target_valid_count}."
        )

    print(
        "Collection complete: "
        f"raw_records={len(raw_records)}, "
        f"valid_preprocessed={len(preprocessed_records)}, "
        f"duplicates_skipped={duplicate_skipped}."
    )

    return raw_records, preprocessed_records[:target_valid_count]


def save_json(records: List[Dict[str, object]], output_path: str) -> None:
    """Save records with UTF-8 encoding and pretty JSON formatting."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for test dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Fetch Bugzilla Core test data and save raw + preprocessed JSON files.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_VALID_SAMPLES,
        help="Number of valid preprocessed records to collect.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for preparing raw_issues.json and preprocessed.json."""
    args = parse_args()
    try:
        raw_records, preprocessed_records = prepare_datasets(target_valid_count=args.target)
        save_json(raw_records, RAW_OUTPUT_FILE)
        save_json(preprocessed_records, PREPROCESSED_OUTPUT_FILE)
    except Exception as exc:
        print(f"Failed to prepare test datasets: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
