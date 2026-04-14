"""
HyDE — Hypothetical Document Embeddings
==========================================
When retrieval confidence is low, HyDE generates a hypothetical
"ideal resolved bug report" using the LLM, embeds it, and re-runs
retrieval to find better matches.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
           Relevance Labels" (2022)
"""

import json
import re
import time

import requests

from config.settings import settings
from modules.embedding import generate_embedding
from modules.retrieval import _dense_retrieval, _sparse_retrieval, _reciprocal_rank_fusion
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_text_block(text: str) -> str:
    """Extract clean text, stripping markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # skip opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _request_ollama(prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
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
            logger.warning(f"HyDE Ollama request failed ({response.status_code}) attempt {attempt + 1}/3")
        except requests.RequestException as exc:
            logger.warning(f"HyDE Ollama request error attempt {attempt + 1}/3: {exc}")
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to get HyDE response from Ollama.")


HYDE_PROMPT = """You are a senior software engineer. Given the following bug report query, write a HYPOTHETICAL detailed resolved bug report that would be the ideal answer document for this query.

The hypothetical document should include:
- A clear description of the bug
- The root cause analysis
- The solution/fix that was applied
- Technical details (components involved, code areas affected)

Write the hypothetical resolved bug report as plain text (no JSON, no markdown headers).
Keep it concise but technically detailed (200-300 words).

QUERY:
{query}

HYPOTHETICAL RESOLVED BUG REPORT:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Use Ollama to generate a hypothetical document that would
    ideally answer the given query. This is the core of HyDE.
    """
    prompt = HYDE_PROMPT.format(query=query[:1000])
    try:
        raw_response = _request_ollama(prompt, temperature=0.3, max_tokens=500)
        hypothetical_doc = _extract_text_block(raw_response)
        if len(hypothetical_doc) < 50:
            logger.warning("HyDE generated a very short hypothetical document, using query as fallback")
            return query
        logger.info(f"HyDE generated hypothetical document ({len(hypothetical_doc)} chars)")
        return hypothetical_doc
    except RuntimeError as e:
        logger.error(f"HyDE generation failed: {e}")
        return query


def hyde_retrieve(
    query: str,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    final_top_k: int = 20,
) -> list[dict]:
    """
    HyDE retrieval pipeline:
    1. Generate hypothetical document from query using LLM
    2. Embed the hypothetical document (not as a query — no instruction prefix)
    3. Run dense retrieval with the hypothetical embedding
    4. Run sparse retrieval with the hypothetical document text
    5. Merge via RRF
    """
    logger.info("HyDE: Generating hypothetical document...")
    hypothetical_doc = generate_hypothetical_document(query)

    logger.info("HyDE: Embedding hypothetical document...")
    hyde_embedding = generate_embedding(hypothetical_doc, is_query=False)

    logger.info("HyDE: Running dense retrieval with hypothetical embedding...")
    dense_results = _dense_retrieval(hyde_embedding.tolist(), top_k=dense_top_k)

    logger.info("HyDE: Running sparse retrieval with hypothetical text...")
    sparse_results = _sparse_retrieval(hypothetical_doc, top_k=sparse_top_k)

    logger.info("HyDE: Fusing results with RRF...")
    fused_results = _reciprocal_rank_fusion(dense_results, sparse_results, k=settings.RRF_K)
    final_results = fused_results[:final_top_k]

    logger.info(
        f"HyDE retrieval complete: {len(final_results)} results "
        f"(from {len(dense_results)} dense + {len(sparse_results)} sparse)"
    )
    return final_results
