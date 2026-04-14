"""
Text Preprocessing Module
===========================
Hybrid preprocessing: regex cleaning + spaCy NLP.
Produces clean unified text for embedding generation.
"""

import json
import re
from typing import Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

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
# Regex Patterns
# ============================================
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
URL_PATTERN = re.compile(r"https?://\S+|ftp://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MENTION_PATTERN = re.compile(r"@[\w-]+")
HEX_CODE_PATTERN = re.compile(r"#[0-9a-fA-F]{3,8}\b")
FILE_PATH_PATTERN = re.compile(r"(?:[/\\][\w.-]+){2,}")
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s*", re.MULTILINE)
EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s.,;:!?'\"-]")
HORIZONTAL_RULE_PATTERN = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
LIST_BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def clean_text_regex(text: str) -> str:
    """Apply regex-based cleaning to remove noise from issue text."""
    if not text:
        return ""
    text = CODE_BLOCK_PATTERN.sub(" ", text)
    text = INLINE_CODE_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = MARKDOWN_HEADER_PATTERN.sub("", text)
    text = HORIZONTAL_RULE_PATTERN.sub(" ", text)
    text = LIST_BULLET_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HEX_CODE_PATTERN.sub(" ", text)
    text = FILE_PATH_PATTERN.sub(" ", text)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = EXCESSIVE_WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def process_with_spacy(text: str, lemmatize: bool = True) -> dict:
    """Process text using spaCy for tokenization, lemmatization, and NER."""
    nlp = _get_nlp()
    doc = nlp(text[:100000])

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if len(token.text) < 2:
            continue
        if lemmatize:
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.text.lower())

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {
        "tokens": tokens,
        "entities": entities,
        "token_count": len(tokens),
    }


def preprocess_issue(issue: dict) -> dict:
    """Full preprocessing pipeline for a single issue."""
    title = issue.get("title", "")
    body = issue.get("body", "")
    labels = issue.get("labels", [])
    comments = issue.get("comments", [])

    clean_title = clean_text_regex(title)
    clean_body = clean_text_regex(body)

    clean_comments = []
    for comment in comments[:3]:
        if isinstance(comment, dict):
            clean_comment = clean_text_regex(comment.get("body", ""))
        else:
            clean_comment = clean_text_regex(str(comment))
        if clean_comment:
            clean_comments.append(clean_comment)

    parts = [clean_title, clean_body]
    if labels:
        parts.append(f"Labels: {' '.join(labels)}")
    if clean_comments:
        parts.append(f"Comments: {' '.join(clean_comments[:2])}")

    unified_text = " . ".join([p for p in parts if p])

    spacy_result = process_with_spacy(unified_text)

    return {
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
        "unified_text": unified_text,
        "tokens": spacy_result["tokens"],
        "entities": spacy_result["entities"],
        "token_count": spacy_result["token_count"],
    }


def preprocess_batch(
    issues: list[dict],
    save_output: bool = True,
    output_path: Optional[str] = None,
) -> list[dict]:
    """Preprocess a batch of issues."""
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
            output_path = str(settings.PROCESSED_ISSUES_FILE)
        settings.ensure_directories()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_issues, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed issues to {output_path}")

    return processed_issues


def load_processed_issues(filepath=None) -> list[dict]:
    """Load previously processed issues from JSON file."""
    if filepath is None:
        filepath = str(settings.PROCESSED_ISSUES_FILE)
    with open(filepath, "r", encoding="utf-8") as f:
        issues = json.load(f)
    logger.info(f"Loaded {len(issues)} processed issues from {filepath}")
    return issues
