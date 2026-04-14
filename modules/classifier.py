"""
Ticket Classification Module (Ollama / llama3.1:8b)
=====================================================
Uses local Ollama LLM to classify tickets into structured categories.

Supports two modes:
  1. **Generic mode** (backward-compatible): classifies into
     [frontend | backend | infrastructure] when no known_teams provided.
  2. **Fine-grained mode**: classifies into a specific team from
     a known_teams list, optionally using retrieved documents as
     context (Retrieval-Augmented Classification).

Classification output:
- type: bug | feature | improvement
- severity: low | medium | high
- team: one of the known teams (or generic bucket)
"""

import json
import re
import time
from typing import Optional

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Generic fallback teams (backward compat)
# ============================================
GENERIC_TEAMS = ["frontend", "backend", "infrastructure"]


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


def _request_ollama(prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
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
    raise RuntimeError("Failed to get response from Ollama. Check service availability.")


# ============================================
# Prompt Templates
# ============================================

# --- Generic prompt (no known_teams) — backward compatible ---
GENERIC_CLASSIFICATION_PROMPT = """You are an expert software issue ticket classification system.

STRICT RULES:
1. Respond with ONLY valid JSON - no markdown, no explanation, no extra text.
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
- "backend": Core engine, file system, language services, debugging, networking
- "infrastructure": Build system, CI/CD, installation, updates, performance, memory

TICKET TO CLASSIFY:
Title: {title}
Description: {description}
Labels: {labels}

Respond with ONLY the JSON object:"""

# --- Fine-grained prompt (with known_teams + retrieved context) ---
FINEGRAINED_CLASSIFICATION_PROMPT = """You are an expert software issue ticket triage system. Your job is to classify a ticket and route it to the correct team.

STRICT RULES:
1. Respond with ONLY valid JSON — no markdown, no explanation, no extra text.
2. The JSON must have exactly these three fields:
   - "type": exactly one of ["bug", "feature", "improvement"]
   - "severity": exactly one of ["low", "medium", "high"]
   - "team": exactly one team name from the VALID TEAMS list below

VALID TEAMS (you MUST pick exactly one):
{valid_teams}

{retrieved_context}

CLASSIFICATION GUIDELINES:
Type:
- "bug": Something is broken, crashing, producing errors, or not working as expected
- "feature": A new capability or functionality that does not exist yet
- "improvement": Enhancement, optimization, or polish to existing functionality

Severity:
- "low": Minor cosmetic issue, nice-to-have, affects few users
- "medium": Functional issue with moderate impact, workaround exists
- "high": Critical bug, crash, data loss, security issue, blocks many users

Team Selection:
- Read the ticket title and description carefully.
- Look at the similar resolved tickets above — they show which teams handled similar issues.
- Pick the team whose past tickets are most semantically similar to this new ticket.
- If multiple teams seem plausible, prefer the team that appears most frequently in the similar tickets.

TICKET TO CLASSIFY:
Title: {title}
Description: {description}
Labels: {labels}

Think step-by-step:
1. What is this ticket about? (core topic / component)
2. Which similar past tickets match most closely?
3. Which team handled those similar tickets?
4. Therefore, the correct team is...

Now respond with ONLY the JSON object:"""


def _format_retrieved_teams_context(retrieved_docs: list[dict]) -> str:
    """Format retrieved docs into a concise context block for the classifier."""
    if not retrieved_docs:
        return ""

    lines = ["SIMILAR RESOLVED TICKETS (use these to determine the correct team):"]
    for i, doc in enumerate(retrieved_docs[:7], 1):
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        title = metadata.get("title", "Unknown")
        team = metadata.get("team", "unknown")
        component = metadata.get("component", "unknown")
        score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
        lines.append(
            f"  #{i}: \"{title}\" → Team: {team} | Component: {component} (relevance: {score:.3f})"
        )
    return "\n".join(lines)


def _extract_retrieved_teams(retrieved_docs: list[dict]) -> list[str]:
    """Extract unique team names from retrieved documents, preserving order."""
    seen = set()
    teams = []
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        team = str(metadata.get("team", "")).strip().lower()
        if team and team != "unknown" and team not in seen:
            seen.add(team)
            teams.append(team)
    return teams


def _find_closest_team(predicted: str, known_teams: list[str], retrieved_docs: list[dict] = None) -> str:
    """
    When the LLM outputs a team not in known_teams, find the best fallback.

    Strategy:
    1. Exact match (case-insensitive) → return it
    2. Substring match (e.g., "html parser" in "dom: html parser") → return it
    3. Fall back to the most frequent team in retrieved docs
    4. Last resort → first known_team
    """
    predicted_lower = predicted.strip().lower()

    # 1. Exact match
    for team in known_teams:
        if team.lower() == predicted_lower:
            return team

    # 2. Substring / partial match
    for team in known_teams:
        team_lower = team.lower()
        if predicted_lower in team_lower or team_lower in predicted_lower:
            return team

    # 3. Most frequent retrieved team that IS in known_teams
    if retrieved_docs:
        retrieved_teams = _extract_retrieved_teams(retrieved_docs)
        for rt in retrieved_teams:
            for team in known_teams:
                if team.lower() == rt.lower():
                    return team

    # 4. Last resort
    return known_teams[0] if known_teams else predicted


def classify_ticket(
    title: str,
    description: str,
    labels: list[str] = None,
    known_teams: list[str] = None,
    retrieved_docs: list[dict] = None,
    max_retries: int = 3,
) -> dict:
    """
    Classify a ticket using local Ollama llama3.1:8b.

    Parameters
    ----------
    title : str
        Ticket title.
    description : str
        Ticket description / body.
    labels : list[str], optional
        Label strings (e.g., ["component:Layout", "team:Tables"]).
    known_teams : list[str], optional
        Valid team names. If provided, the classifier is constrained
        to output one of these teams (Fine-grained mode).
        If None, falls back to generic 3-team classification.
    retrieved_docs : list[dict], optional
        Top reranked documents from retrieval. Used to provide
        context about which teams handled similar tickets
        (Retrieval-Augmented Classification).
    max_retries : int
        Number of LLM call retries on failure.

    Returns
    -------
    dict with keys: type, severity, team
    """
    labels_str = ", ".join(labels) if labels else "none"
    use_finegrained = known_teams is not None and len(known_teams) > 0

    if use_finegrained:
        # --- Fine-grained mode ---
        valid_teams_str = "\n".join(f"  - {team}" for team in known_teams)
        retrieved_context = _format_retrieved_teams_context(retrieved_docs or [])

        prompt = FINEGRAINED_CLASSIFICATION_PROMPT.format(
            title=title,
            description=description[:2000],
            labels=labels_str,
            valid_teams=valid_teams_str,
            retrieved_context=retrieved_context,
        )
    else:
        # --- Generic mode (backward compatible) ---
        prompt = GENERIC_CLASSIFICATION_PROMPT.format(
            title=title,
            description=description[:2000],
            labels=labels_str,
        )

    valid_types = {"bug", "feature", "improvement"}
    valid_severities = {"low", "medium", "high"}
    valid_team_set = {t.lower() for t in known_teams} if use_finegrained else {"frontend", "backend", "infrastructure"}

    for attempt in range(max_retries):
        try:
            response_text = _request_ollama(prompt, temperature=0.1, max_tokens=300)
            json_block = _extract_json_block(response_text)
            result = json.loads(json_block)

            # --- Validate type ---
            if result.get("type") not in valid_types:
                result["type"] = "bug"

            # --- Validate severity ---
            if result.get("severity") not in valid_severities:
                result["severity"] = "medium"

            # --- Validate team ---
            predicted_team = str(result.get("team", "")).strip().lower()
            if use_finegrained:
                if predicted_team not in valid_team_set:
                    # LLM picked a team not in the list — find closest match
                    resolved_team = _find_closest_team(predicted_team, known_teams, retrieved_docs)
                    logger.info(
                        f"Classifier output '{predicted_team}' not in known_teams, "
                        f"resolved to '{resolved_team}'"
                    )
                    result["team"] = resolved_team
                else:
                    # Normalize to the canonical casing from known_teams
                    for kt in known_teams:
                        if kt.lower() == predicted_team:
                            result["team"] = kt
                            break
            else:
                if predicted_team not in valid_team_set:
                    result["team"] = "backend"

            logger.info(
                f"Classification: type={result['type']}, "
                f"severity={result['severity']}, team={result['team']}"
            )
            return result

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"Classification attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # --- All retries failed: smart fallback ---
    logger.warning("All classification attempts failed. Using fallback.")
    fallback_team = "backend"
    if use_finegrained and retrieved_docs:
        retrieved_teams = _extract_retrieved_teams(retrieved_docs)
        if retrieved_teams:
            fallback_team = retrieved_teams[0]
            # Ensure it's in known_teams
            fallback_team = _find_closest_team(fallback_team, known_teams, retrieved_docs)
    elif use_finegrained and known_teams:
        fallback_team = known_teams[0]

    return {"type": "bug", "severity": "medium", "team": fallback_team}
