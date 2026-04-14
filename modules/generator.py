"""
RAG Response Generation Module (Ollama / llama3.1:8b)
======================================================
Generates debugging suggestions and routing explanations using
local Ollama LLM with Retrieval-Augmented Generation (RAG).
"""

import json
import re
import time

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_json_block(text: str) -> str:
    """Extract a JSON object from mixed LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0)
    return "{}"


def _request_ollama(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """Call Ollama /api/generate with retry logic."""
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
            if response.status_code == 200:
                body = response.json()
                return str(body.get("response", "")).strip()
            logger.warning(f"Ollama request failed ({response.status_code}) attempt {attempt + 1}/3")
        except requests.RequestException as exc:
            logger.warning(f"Ollama request error attempt {attempt + 1}/3: {exc}")
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to get response from Ollama.")


GENERATION_PROMPT = """You are a senior software engineer specializing in debugging and issue triage.
You have been given an original software issue ticket along with similar past issues that were previously resolved.

STRICT RULES:
1. Respond with ONLY valid JSON - no markdown, no explanation, no extra text.
2. The JSON must have exactly these fields:
   - "routing_explanation": A brief explanation of why this ticket should go to the assigned team (2-3 sentences)
   - "debugging_steps": An array of 3-5 debugging steps (strings)
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
    """Format retrieved documents into a readable context string."""
    if not retrieved_docs:
        return "No similar past issues found."
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "Unknown")
        issue_num = metadata.get("issue_number", "N/A")
        score = doc.get("rerank_score", doc.get("rrf_score", 0))
        doc_text = doc.get("document", "")[:500]
        context_parts.append(
            f"--- Similar Issue #{i} (Issue #{issue_num}, Relevance: {score:.4f}) ---\n"
            f"Title: {title}\n"
            f"Content: {doc_text}\n"
        )
    return "\n".join(context_parts)


def generate_response(
    title: str,
    description: str,
    labels: list[str] = None,
    classification: dict = None,
    retrieved_docs: list[dict] = None,
    max_retries: int = 3,
) -> dict:
    """Generate debugging suggestions using Ollama with RAG context."""
    labels_str = ", ".join(labels) if labels else "none"
    classification = classification or {"type": "unknown", "severity": "unknown", "team": "unknown"}

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
            response_text = _request_ollama(prompt, temperature=0.2, max_tokens=700)
            json_block = _extract_json_block(response_text)
            result = json.loads(json_block)

            if "routing_explanation" not in result:
                result["routing_explanation"] = "Routing based on ticket content analysis."
            if "debugging_steps" not in result:
                result["debugging_steps"] = ["Review the issue description for more context"]
            if "possible_causes" not in result:
                result["possible_causes"] = ["Further investigation needed"]

            if isinstance(result["debugging_steps"], str):
                result["debugging_steps"] = [result["debugging_steps"]]
            if isinstance(result["possible_causes"], str):
                result["possible_causes"] = [result["possible_causes"]]

            logger.info(
                f"Generated response with {len(result['debugging_steps'])} debugging steps "
                f"and {len(result['possible_causes'])} possible causes"
            )
            return result

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

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
