"""
RAG Response Generation Module (Gemini)
==========================================
Generates debugging suggestions and routing explanations using
Google Gemini with Retrieval-Augmented Generation (RAG).

The prompt includes the original ticket plus retrieved and reranked
similar issues as context, enabling the model to ground its response
in real historical data (true RAG behavior).

Key Design:
- Retrieved similar issues provide grounding context
- Structured JSON output with debugging steps and root causes
- Temperature kept low for consistency
- Prompt clearly separates original ticket from retrieved context
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
# Gemini Client (singleton, shared with classifier)
# ============================================
_client = None


def _get_client():
    """Get or create the Gemini API client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        logger.info(f"Gemini generator client initialized (model: {settings.GEMINI_MODEL})")
    return _client


# ============================================
# Generation Prompt Template
# ============================================
GENERATION_PROMPT = """You are a senior software engineer specializing in debugging and issue triage. 
You have been given an original software issue ticket along with similar past issues that were previously resolved.

Use the similar past issues as CONTEXT to provide more accurate and grounded suggestions.

STRICT RULES:
1. Respond with ONLY valid JSON - no markdown, no explanation, no extra text.
2. The JSON must have exactly these fields:
   - "routing_explanation": A brief explanation of why this ticket should go to the assigned team (2-3 sentences)
   - "debugging_steps": An array of 3-5 numbered debugging steps (strings)
   - "possible_causes": An array of 2-4 possible root causes (strings)

ORIGINAL TICKET:
Title: {title}
Description: {description}
Labels: {labels}
Classification: Type={type}, Severity={severity}, Team={team}

SIMILAR PAST ISSUES (for context):
{retrieved_context}

Based on the original ticket and the patterns from similar past issues, generate your response.
Respond with ONLY the JSON object:"""


def _format_retrieved_context(retrieved_docs: list[dict]) -> str:
    """
    Format retrieved documents into a readable context string for the prompt.

    Args:
        retrieved_docs: List of retrieved document dictionaries.

    Returns:
        Formatted context string.
    """
    if not retrieved_docs:
        return "No similar past issues found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "Unknown")
        issue_num = metadata.get("issue_number", "N/A")
        labels = metadata.get("labels", "")
        score = doc.get("rerank_score", doc.get("rrf_score", 0))

        # Get document text (truncated)
        doc_text = doc.get("document", "")[:500]

        context_parts.append(
            f"--- Similar Issue #{i} (Issue #{issue_num}, Relevance: {score:.4f}) ---\n"
            f"Title: {title}\n"
            f"Labels: {labels}\n"
            f"Content: {doc_text}\n"
        )

    return "\n".join(context_parts)


def _parse_generation_response(response_text: str) -> Optional[dict]:
    """
    Parse the Gemini generation response into a structured dictionary.

    Args:
        response_text: Raw response from Gemini.

    Returns:
        Parsed response dict or None if parsing fails.
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

        # Validate and ensure required fields
        if "routing_explanation" not in result:
            result["routing_explanation"] = "Routing based on ticket content analysis."
        if "debugging_steps" not in result:
            result["debugging_steps"] = ["Review the issue description for more context"]
        if "possible_causes" not in result:
            result["possible_causes"] = ["Further investigation needed"]

        # Ensure lists are actually lists
        if isinstance(result["debugging_steps"], str):
            result["debugging_steps"] = [result["debugging_steps"]]
        if isinstance(result["possible_causes"], str):
            result["possible_causes"] = [result["possible_causes"]]

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse generation JSON: {e}")
        logger.warning(f"Raw response: {text[:300]}")
        return None


def generate_response(
    title: str,
    description: str,
    labels: list[str] = None,
    classification: dict = None,
    retrieved_docs: list[dict] = None,
    max_retries: int = 3,
) -> dict:
    """
    Generate debugging suggestions and routing explanation using
    Gemini with RAG context from retrieved similar issues.

    Args:
        title: Ticket title.
        description: Ticket description (cleaned).
        labels: Optional list of labels.
        classification: Classification result dict (type, severity, team).
        retrieved_docs: Reranked retrieval results for context.
        max_retries: Number of retry attempts.

    Returns:
        Generated response dictionary with routing_explanation,
        debugging_steps, and possible_causes.
    """
    client = _get_client()

    labels_str = ", ".join(labels) if labels else "none"
    classification = classification or {"type": "unknown", "severity": "unknown", "team": "unknown"}

    # Format retrieved context
    retrieved_context = _format_retrieved_context(retrieved_docs or [])

    prompt = GENERATION_PROMPT.format(
        title=title,
        description=description[:2000],
        labels=labels_str,
        type=classification.get("type", "unknown"),
        severity=classification.get("severity", "unknown"),
        team=classification.get("team", "unknown"),
        retrieved_context=retrieved_context,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Slightly higher for more creative suggestions
                    max_output_tokens=1000,
                ),
            )

            response_text = response.text
            result = _parse_generation_response(response_text)

            if result is not None:
                logger.info(
                    f"Generated response with {len(result['debugging_steps'])} debugging steps "
                    f"and {len(result['possible_causes'])} possible causes"
                )
                return result

            logger.warning(
                f"Generation attempt {attempt + 1} produced invalid JSON. Retrying..."
            )

        except Exception as e:
            logger.error(f"Generation error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # Fallback response
    logger.warning("All generation attempts failed. Using fallback response.")
    return {
        "routing_explanation": (
            f"This ticket appears to be a {classification.get('type', 'bug')} "
            f"with {classification.get('severity', 'medium')} severity, "
            f"routed to the {classification.get('team', 'backend')} team."
        ),
        "debugging_steps": [
            "1. Review the issue description and reproduce the problem",
            "2. Check recent code changes in the related component",
            "3. Examine error logs for relevant stack traces",
            "4. Test with different configurations to isolate the cause",
        ],
        "possible_causes": [
            "Code regression from recent changes",
            "Edge case not covered by existing tests",
        ],
    }
