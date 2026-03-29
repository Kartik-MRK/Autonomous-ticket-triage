"""
Text Preprocessing Module
===========================
Implements a hybrid preprocessing approach combining:
1. Regex-based cleaning: Remove URLs, code blocks, HTML, mentions, noise
2. spaCy-based NLP: Tokenization, lemmatization, stop word removal, NER

Produces a clean unified text representation from issue title, body, labels,
and comments suitable for embedding generation.

Design Notes:
- spaCy model loaded lazily (only when needed) to reduce startup time
- Processing is deterministic and reproducible
- Supports batch processing for efficiency
"""

import json
import re
from typing import Optional

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Lazy spaCy model loading
# ============================================
_nlp = None


def _get_nlp():
    """Lazily load the spaCy English model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            raise
    return _nlp


# ============================================
# Regex Patterns for Text Cleaning
# ============================================
# Multi-line code blocks (```...```)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Inline code (`...`)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

# URLs (http, https, ftp)
URL_PATTERN = re.compile(r"https?://\S+|ftp://\S+|www\.\S+")

# HTML tags
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# GitHub mentions (@username)
MENTION_PATTERN = re.compile(r"@[\w-]+")

# Hex color codes (#abc123)
HEX_CODE_PATTERN = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# File paths (Unix and Windows style)
FILE_PATH_PATTERN = re.compile(r"(?:[/\\][\w.-]+){2,}")

# Markdown image/link syntax
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")

# Markdown headers (#, ##, ###, etc.)
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s*", re.MULTILINE)

# Multiple whitespace/newlines
EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"\s+")

# Special characters and emoji-like sequences
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s.,;:!?'\"-]")

# Markdown horizontal rules and separators
HORIZONTAL_RULE_PATTERN = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)

# Markdown list bullets
LIST_BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def clean_text_regex(text: str) -> str:
    """
    Apply regex-based cleaning to remove noise from issue text.
    Order of operations matters - code blocks first, then finer patterns.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove code blocks first (they may contain URLs, paths, etc.)
    text = CODE_BLOCK_PATTERN.sub(" ", text)
    text = INLINE_CODE_PATTERN.sub(" ", text)

    # Remove URLs
    text = URL_PATTERN.sub(" ", text)

    # Remove HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Remove markdown elements
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)  # Keep link text
    text = MARKDOWN_HEADER_PATTERN.sub("", text)
    text = HORIZONTAL_RULE_PATTERN.sub(" ", text)
    text = LIST_BULLET_PATTERN.sub("", text)

    # Remove mentions, hex codes, file paths
    text = MENTION_PATTERN.sub(" ", text)
    text = HEX_CODE_PATTERN.sub(" ", text)
    text = FILE_PATH_PATTERN.sub(" ", text)

    # Remove remaining special characters (keep basic punctuation)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = EXCESSIVE_WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def process_with_spacy(text: str, lemmatize: bool = True) -> dict:
    """
    Process text using spaCy for tokenization, lemmatization, and NER.

    Args:
        text: Cleaned text to process.
        lemmatize: Whether to apply lemmatization.

    Returns:
        Dictionary with processed tokens and extracted entities.
    """
    nlp = _get_nlp()

    # Process with spaCy (limit text length for efficiency)
    doc = nlp(text[:100000])  # spaCy has memory limits on very long texts

    # Tokenize and optionally lemmatize
    tokens = []
    for token in doc:
        # Skip stop words, punctuation, and whitespace
        if token.is_stop or token.is_punct or token.is_space:
            continue
        # Skip very short tokens
        if len(token.text) < 2:
            continue

        if lemmatize:
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.text.lower())

    # Extract named entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
        })

    return {
        "tokens": tokens,
        "entities": entities,
        "token_count": len(tokens),
    }


def preprocess_issue(issue: dict) -> dict:
    """
    Full preprocessing pipeline for a single issue.
    Combines regex cleaning and spaCy processing.

    Args:
        issue: Raw issue dictionary with title, body, labels, comments.

    Returns:
        Preprocessed issue dictionary with original and cleaned fields.
    """
    title = issue.get("title", "")
    body = issue.get("body", "")
    labels = issue.get("labels", [])
    comments = issue.get("comments", [])

    # ---- Step 1: Regex cleaning ----
    clean_title = clean_text_regex(title)
    clean_body = clean_text_regex(body)

    # Clean comment bodies
    clean_comments = []
    for comment in comments[:3]:  # Limit to first 3 comments
        if isinstance(comment, dict):
            clean_comment = clean_text_regex(comment.get("body", ""))
        else:
            clean_comment = clean_text_regex(str(comment))
        if clean_comment:
            clean_comments.append(clean_comment)

    # ---- Step 2: Build unified text ----
    # Combine title, body, labels, and key comments into one text block
    parts = [clean_title, clean_body]

    if labels:
        label_text = " ".join(labels)
        parts.append(f"Labels: {label_text}")

    if clean_comments:
        comment_text = " ".join(clean_comments[:2])  # Use top 2 comments
        parts.append(f"Comments: {comment_text}")

    unified_text = " . ".join([p for p in parts if p])

    # ---- Step 3: spaCy processing ----
    spacy_result = process_with_spacy(unified_text)

    # Build the processed issue
    processed = {
        "number": issue.get("number"),
        "original_title": title,
        "original_body": body,
        "clean_title": clean_title,
        "clean_body": clean_body,
        "labels": labels,
        "assignees": issue.get("assignees", []),
        "state": issue.get("state", ""),
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        # Unified clean text for embedding
        "unified_text": unified_text,
        # spaCy processed tokens (for BM25)
        "tokens": spacy_result["tokens"],
        "entities": spacy_result["entities"],
        "token_count": spacy_result["token_count"],
    }

    return processed


def preprocess_batch(
    issues: list[dict],
    save_output: bool = True,
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Preprocess a batch of issues.

    Args:
        issues: List of raw issue dictionaries.
        save_output: Whether to save processed data to disk.
        output_path: Custom output path. Defaults to settings.PROCESSED_ISSUES_FILE.

    Returns:
        List of preprocessed issue dictionaries.
    """
    logger.info(f"Starting preprocessing of {len(issues)} issues...")

    processed_issues = []
    for i, issue in enumerate(issues):
        try:
            processed = preprocess_issue(issue)
            processed_issues.append(processed)

            if (i + 1) % 50 == 0:
                logger.info(f"Preprocessed {i + 1}/{len(issues)} issues")

        except Exception as e:
            logger.warning(f"Failed to preprocess issue #{issue.get('number', 'unknown')}: {e}")
            continue

    logger.info(f"Preprocessing complete: {len(processed_issues)}/{len(issues)} issues processed")

    if save_output:
        if output_path is None:
            output_path = settings.PROCESSED_ISSUES_FILE

        settings.ensure_directories()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_issues, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed issues to {output_path}")

    return processed_issues


def load_processed_issues(filepath=None) -> list[dict]:
    """
    Load previously processed issues from JSON file.

    Args:
        filepath: Path to the processed JSON file.

    Returns:
        List of preprocessed issue dictionaries.
    """
    if filepath is None:
        filepath = settings.PROCESSED_ISSUES_FILE

    with open(filepath, "r", encoding="utf-8") as f:
        issues = json.load(f)

    logger.info(f"Loaded {len(issues)} processed issues from {filepath}")
    return issues
