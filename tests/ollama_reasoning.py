"""Ollama-based classification and response-generation helpers for tests."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv


TESTS_DIR = Path(__file__).resolve().parent
load_dotenv(TESTS_DIR / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
MAX_RETRIES = 3


def _extract_json_block(text: str) -> str:
    """Extract a JSON object from plain text, fenced markdown, or mixed output."""
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
    """Call Ollama /api/generate with simple retry logic."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                body = response.json()
                return str(body.get("response", "")).strip()

            print(
                f"Ollama request failed ({response.status_code}) attempt "
                f"{attempt}/{MAX_RETRIES}: {response.text[:200]}"
            )
        except requests.RequestException as exc:
            print(f"Ollama request error attempt {attempt}/{MAX_RETRIES}: {exc}")

        if attempt < MAX_RETRIES:
            backoff = 2 ** (attempt - 1)
            time.sleep(backoff)

    raise RuntimeError(
        "Failed to get a valid response from Ollama. "
        "Check OLLAMA_BASE_URL, OLLAMA_MODEL, and service availability."
    )


def classify_issue_with_ollama(
    query_text: str,
    component_hint: str,
    candidate_teams: List[str],
) -> Dict[str, object]:
    """Classify ticket and return ranked team candidates via Ollama."""
    team_list = [team.strip() for team in candidate_teams if team and team.strip()]
    if not team_list:
        team_list = ["general"]

    prompt = f"""
You are a software triage classifier.
Return ONLY JSON with this exact schema:
{{
  "type": "bug|feature|improvement",
  "severity": "low|medium|high",
  "top_teams": ["team1", "team2", "team3", "team4", "team5"],
  "rationale": "one short sentence"
}}

Rules:
- top_teams must be chosen ONLY from this candidate list: {team_list}
- Order top_teams from most likely to least likely.
- Return up to 5 teams, no duplicates.

Ticket text:
{query_text}

Component hint:
{component_hint}
""".strip()

    raw = _request_ollama(prompt, temperature=0.1, max_tokens=300)
    json_block = _extract_json_block(raw)

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        parsed = {}

    ticket_type = str(parsed.get("type", "bug")).strip().lower()
    if ticket_type not in {"bug", "feature", "improvement"}:
        ticket_type = "bug"

    severity = str(parsed.get("severity", "medium")).strip().lower()
    if severity not in {"low", "medium", "high"}:
        severity = "medium"

    model_teams = parsed.get("top_teams", [])
    if isinstance(model_teams, str):
        model_teams = [model_teams]

    normalized_candidates = {team.lower(): team for team in team_list}
    ranked_teams: List[str] = []
    for team in model_teams:
        key = str(team).strip().lower()
        if key in normalized_candidates and normalized_candidates[key] not in ranked_teams:
            ranked_teams.append(normalized_candidates[key])

    for team in team_list:
        if len(ranked_teams) >= 5:
            break
        if team not in ranked_teams:
            ranked_teams.append(team)

    rationale = str(parsed.get("rationale", "Classification derived from ticket text and retrieved context.")).strip()

    return {
        "type": ticket_type,
        "severity": severity,
        "top_teams": ranked_teams[:5],
        "rationale": rationale,
    }


def generate_ticket_response_with_ollama(
    query_text: str,
    classification: Dict[str, object],
    retrieved_docs: List[Dict[str, object]],
) -> Dict[str, object]:
    """Generate routing and debugging response in JSON using Ollama."""
    context_parts: List[str] = []
    for idx, doc in enumerate(retrieved_docs[:5], start=1):
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        snippet = str(doc.get("document", ""))[:320]
        context_parts.append(
            f"[{idx}] id={doc.get('id', 'n/a')} "
            f"title={metadata.get('title', 'unknown')} "
            f"team={metadata.get('team', 'unknown')} "
            f"component={metadata.get('component', 'unknown')} "
            f"score={doc.get('rerank_score', doc.get('rrf_score', 0))}\n"
            f"snippet={snippet}"
        )

    retrieved_context = "\n\n".join(context_parts) if context_parts else "No retrieval context available."

    prompt = f"""
You are a senior software triage assistant.
Return ONLY JSON with this exact schema:
{{
  "routing_explanation": "2-3 sentences",
  "debugging_steps": ["step 1", "step 2", "step 3"],
  "possible_causes": ["cause 1", "cause 2"]
}}

Ticket:
{query_text}

Classification hint:
type={classification.get('type', 'bug')},
severity={classification.get('severity', 'medium')},
team={classification.get('top_teams', ['general'])[0] if classification.get('top_teams') else 'general'}

Retrieved context:
{retrieved_context}
""".strip()

    raw = _request_ollama(prompt, temperature=0.2, max_tokens=600)
    json_block = _extract_json_block(raw)

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        parsed = {}

    routing_explanation = str(
        parsed.get(
            "routing_explanation",
            "The issue is routed based on its technical scope and retrieved similar incidents.",
        )
    ).strip()

    debugging_steps = parsed.get("debugging_steps", [])
    if isinstance(debugging_steps, str):
        debugging_steps = [debugging_steps]
    debugging_steps = [str(step).strip() for step in debugging_steps if str(step).strip()]
    if not debugging_steps:
        debugging_steps = [
            "Reproduce the issue and capture logs.",
            "Compare with similar incidents from retrieval context.",
            "Validate the suspected fix with targeted tests.",
        ]

    possible_causes = parsed.get("possible_causes", [])
    if isinstance(possible_causes, str):
        possible_causes = [possible_causes]
    possible_causes = [str(cause).strip() for cause in possible_causes if str(cause).strip()]
    if not possible_causes:
        possible_causes = [
            "Regression introduced by recent code changes.",
            "Edge-case input handling missing in affected component.",
        ]

    return {
        "routing_explanation": routing_explanation,
        "debugging_steps": debugging_steps[:5],
        "possible_causes": possible_causes[:4],
    }
