"""
Data Ingestion Module
======================
Fetches issues from GitHub repositories using the GraphQL API (v4).
Handles cursor-based pagination, rate limiting with exponential backoff,
and filters out invalid or incomplete issues.

Key Design Decisions:
- Uses GraphQL (not REST) for efficient field selection and single-request data fetching
- Fetches: title, body, labels, assignees, createdAt, updatedAt, state, number, comments
- Implements exponential backoff for rate limit handling
- Filters issues missing title or body
"""

import json
import time
from typing import Optional

import requests

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# GraphQL Query Template
# ============================================
ISSUES_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        state
        createdAt
        updatedAt
        labels(first: 10) {
          nodes {
            name
          }
        }
        assignees(first: 5) {
          nodes {
            login
          }
        }
        comments(first: 5) {
          nodes {
            body
            createdAt
            author {
              login
            }
          }
        }
      }
    }
  }
}
"""


def _make_graphql_request(
    query: str,
    variables: dict,
    max_retries: int = 5
) -> dict:
    """
    Execute a GraphQL request with exponential backoff retry logic.

    Args:
        query: GraphQL query string.
        variables: Query variables dictionary.
        max_retries: Maximum number of retry attempts.

    Returns:
        Parsed JSON response data.

    Raises:
        Exception: If all retries are exhausted.
    """
    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                settings.GITHUB_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=30,
            )

            # Handle rate limiting
            if response.status_code == 403:
                reset_time = response.headers.get("X-RateLimit-Reset")
                if reset_time:
                    wait_seconds = max(int(reset_time) - int(time.time()), 1)
                    logger.warning(f"Rate limited. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    wait_time = 2 ** attempt * 5
                    logger.warning(f"Rate limited (no reset header). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # Handle server errors
            if response.status_code >= 500:
                wait_time = 2 ** attempt * 2
                logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            # Check for GraphQL-level errors
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                raise Exception(f"GraphQL errors: {data['errors']}")

            return data

        except requests.exceptions.Timeout:
            wait_time = 2 ** attempt * 2
            logger.warning(f"Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except requests.exceptions.ConnectionError:
            wait_time = 2 ** attempt * 2
            logger.warning(f"Connection error. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    raise Exception(f"Failed to fetch data after {max_retries} retries")


def _parse_issue(raw_issue: dict) -> Optional[dict]:
    """
    Parse a raw GraphQL issue node into a clean dictionary.
    Returns None if the issue is invalid (missing title or body).

    Args:
        raw_issue: Raw issue node from GraphQL response.

    Returns:
        Parsed issue dictionary or None if invalid.
    """
    title = raw_issue.get("title", "").strip()
    body = raw_issue.get("body", "").strip() if raw_issue.get("body") else ""

    # Filter out issues with missing title or body
    if not title or not body:
        return None

    # Filter out very short bodies (likely not useful)
    if len(body) < 20:
        return None

    # Extract labels
    labels = [
        label["name"]
        for label in raw_issue.get("labels", {}).get("nodes", [])
    ]

    # Extract assignees
    assignees = [
        assignee["login"]
        for assignee in raw_issue.get("assignees", {}).get("nodes", [])
    ]

    # Extract comments
    comments = []
    for comment in raw_issue.get("comments", {}).get("nodes", []):
        comment_body = comment.get("body", "").strip()
        if comment_body:
            comments.append({
                "body": comment_body,
                "author": comment.get("author", {}).get("login", "unknown") if comment.get("author") else "unknown",
                "created_at": comment.get("createdAt", ""),
            })

    return {
        "number": raw_issue.get("number"),
        "title": title,
        "body": body,
        "state": raw_issue.get("state", "UNKNOWN"),
        "labels": labels,
        "assignees": assignees,
        "created_at": raw_issue.get("createdAt", ""),
        "updated_at": raw_issue.get("updatedAt", ""),
        "comments": comments,
    }


def fetch_issues(
    max_issues: Optional[int] = None,
    page_size: int = 50,
) -> list[dict]:
    """
    Fetch issues from the configured GitHub repository using GraphQL API.
    Implements cursor-based pagination and filters invalid issues.

    Args:
        max_issues: Maximum number of valid issues to fetch. Defaults to settings.MAX_ISSUES.
        page_size: Number of issues per GraphQL request (max 100).

    Returns:
        List of parsed issue dictionaries.
    """
    if max_issues is None:
        max_issues = settings.MAX_ISSUES

    page_size = min(page_size, 100)  # GitHub GraphQL limit

    logger.info(
        f"Starting issue ingestion from {settings.GITHUB_REPO_OWNER}/{settings.GITHUB_REPO_NAME} "
        f"(target: {max_issues} issues)"
    )

    all_issues = []
    cursor = None
    total_fetched = 0
    total_filtered = 0

    while len(all_issues) < max_issues:
        # Calculate how many to request this batch
        remaining = max_issues - len(all_issues)
        batch_size = min(page_size, remaining + 10)  # Fetch extra to account for filtering

        variables = {
            "owner": settings.GITHUB_REPO_OWNER,
            "name": settings.GITHUB_REPO_NAME,
            "first": batch_size,
            "after": cursor,
        }

        logger.info(f"Fetching batch of {batch_size} issues (cursor: {cursor})...")
        data = _make_graphql_request(ISSUES_QUERY, variables)

        issues_data = data["data"]["repository"]["issues"]
        page_info = issues_data["pageInfo"]
        raw_issues = issues_data["nodes"]

        total_fetched += len(raw_issues)

        # Parse and filter issues
        for raw_issue in raw_issues:
            parsed = _parse_issue(raw_issue)
            if parsed is not None:
                all_issues.append(parsed)
                if len(all_issues) >= max_issues:
                    break
            else:
                total_filtered += 1

        logger.info(
            f"Progress: {len(all_issues)}/{max_issues} valid issues "
            f"({total_filtered} filtered, {total_fetched} total fetched)"
        )

        # Check if we've reached the end
        if not page_info["hasNextPage"]:
            logger.info("Reached end of available issues")
            break

        cursor = page_info["endCursor"]

        # Small delay between requests to be polite to the API
        time.sleep(0.5)

    logger.info(
        f"Ingestion complete: {len(all_issues)} valid issues "
        f"({total_filtered} filtered out of {total_fetched} total)"
    )

    return all_issues


def save_issues(issues: list[dict], filepath=None) -> str:
    """
    Save fetched issues to a JSON file.

    Args:
        issues: List of parsed issue dictionaries.
        filepath: Output file path. Defaults to settings.RAW_ISSUES_FILE.

    Returns:
        Path to the saved file.
    """
    if filepath is None:
        filepath = settings.RAW_ISSUES_FILE

    settings.ensure_directories()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(issues)} issues to {filepath}")
    return str(filepath)


def load_raw_issues(filepath=None) -> list[dict]:
    """
    Load previously saved raw issues from JSON file.

    Args:
        filepath: Path to the JSON file. Defaults to settings.RAW_ISSUES_FILE.

    Returns:
        List of issue dictionaries.
    """
    if filepath is None:
        filepath = settings.RAW_ISSUES_FILE

    with open(filepath, "r", encoding="utf-8") as f:
        issues = json.load(f)

    logger.info(f"Loaded {len(issues)} raw issues from {filepath}")
    return issues
