"""
Ticket Classification Module (Gemini)
========================================
Uses Google AI Studio (Gemini) to classify tickets into structured categories.

Classification output:
- type: bug | feature | improvement
- severity: low | medium | high
- team: frontend | backend | infrastructure

Key Design Decisions:
- Strict prompt engineering forces JSON-only output
- Temperature=0 for deterministic, constrained responses
- Retry logic for malformed JSON responses
- Fallback defaults for robustness
- Model name configurable via settings for easy switching

References:
- https://ai.google.dev/api
"""

import json
import time
from typing import Optional

from google import genai
from google.genai import types

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Gemini Client (singleton)
# ============================================
_client = None


def _get_client():
    """Get or create the Gemini API client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        logger.info(f"Gemini client initialized (model: {settings.GEMINI_MODEL})")
    return _client


# ============================================
# Classification Prompt Template
# ============================================
CLASSIFICATION_PROMPT = """You are an expert software issue ticket classification system. Your task is to analyze the given ticket and classify it precisely.

STRICT RULES:
1. You MUST respond with ONLY valid JSON - no markdown, no explanation, no extra text.
2. The JSON must have exactly these three fields:
   - "type": exactly one of ["bug", "feature", "improvement"]
   - "severity": exactly one of ["low", "medium", "high"]
   - "team": exactly one of ["frontend", "backend", "infrastructure"]

CLASSIFICATION GUIDELINES:
- "bug": Something is broken, crashing, or not working as expected
- "feature": A new capability or functionality request
- "improvement": Enhancement to existing functionality, performance, UX

- "low": Minor cosmetic issue, nice-to-have, affects few users
- "medium": Functional issue, moderate impact, workaround exists
- "high": Critical bug, data loss, crash, security issue, affects many users

- "frontend": UI, editor, themes, keybindings, extensions UI, rendering
- "backend": Core engine, file system, language services, debugging, Git integration
- "infrastructure": Build system, CI/CD, installation, updates, performance, memory

TICKET TO CLASSIFY:
Title: {title}
Description: {description}
Labels: {labels}

Respond with ONLY the JSON object:"""


def _parse_classification_response(response_text: str) -> Optional[dict]:
    """
    Parse the Gemini response into a classification dictionary.
    Handles common response formatting issues.

    Args:
        response_text: Raw response from Gemini.

    Returns:
        Parsed classification dict or None if parsing fails.
    """
    text = response_text.strip()

    # Remove markdown code block markers if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)

        # Validate required fields
        valid_types = {"bug", "feature", "improvement"}
        valid_severities = {"low", "medium", "high"}
        valid_teams = {"frontend", "backend", "infrastructure"}

        if result.get("type") not in valid_types:
            result["type"] = "bug"  # Safe default
        if result.get("severity") not in valid_severities:
            result["severity"] = "medium"
        if result.get("team") not in valid_teams:
            result["team"] = "backend"

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classification JSON: {e}")
        logger.warning(f"Raw response: {text[:200]}")
        return None


def classify_ticket(
    title: str,
    description: str,
    labels: list[str] = None,
    max_retries: int = 3,
) -> dict:
    """
    Classify a ticket using Google Gemini.

    Args:
        title: Ticket title.
        description: Ticket description (cleaned).
        labels: Optional list of existing labels.
        max_retries: Number of retry attempts for malformed responses.

    Returns:
        Classification dictionary with type, severity, and team.
    """
    client = _get_client()

    labels_str = ", ".join(labels) if labels else "none"

    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        description=description[:2000],  # Limit description length
        labels=labels_str,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=200,
                ),
            )

            response_text = response.text
            result = _parse_classification_response(response_text)

            if result is not None:
                logger.info(
                    f"Classification: type={result['type']}, "
                    f"severity={result['severity']}, team={result['team']}"
                )
                return result

            logger.warning(
                f"Classification attempt {attempt + 1} produced invalid JSON. Retrying..."
            )

        except Exception as e:
            logger.error(f"Classification error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # Fallback if all retries fail
    logger.warning("All classification attempts failed. Using defaults.")
    return {
        "type": "bug",
        "severity": "medium",
        "team": "backend",
    }
