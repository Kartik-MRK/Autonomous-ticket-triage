"""
Triage Pipeline — End-to-End Orchestration
=============================================
Integrates all modules into a sequential pipeline:
1. Preprocess ticket text (regex + spaCy)
2. Hybrid retrieval (dense + BM25 + RRF)
3. Cross-encoder reranking (BAAI/bge-reranker-base)
4. Retrieval-Augmented Classification (fine-grained teams)
5. HyDE fallback if classifier-retrieval agreement is low
6. RAG response generation (Ollama llama3.1:8b)
"""

import time
from typing import Optional

from modules.preprocessing import preprocess_issue
from modules.retrieval import hybrid_retrieve
from modules.reranker import rerank, get_top_confidence
from modules.hyde import hyde_retrieve
from modules.classifier import classify_ticket
from modules.generator import generate_response
from modules.vector_store import get_all_documents
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ---- Cached corpus team list (extracted once at first use) ----
_cached_known_teams: list[str] | None = None


def _get_known_teams() -> list[str]:
    """Extract unique team names from the ChromaDB corpus. Cached."""
    global _cached_known_teams
    if _cached_known_teams is not None:
        return _cached_known_teams

    try:
        all_docs = get_all_documents()
        corpus_metadatas = all_docs.get("metadatas", []) or []
        seen = set()
        teams = []
        for meta in corpus_metadatas:
            if isinstance(meta, dict):
                team = str(meta.get("team", "")).strip().lower()
                if team and team != "unknown" and team not in seen:
                    seen.add(team)
                    teams.append(team)
        _cached_known_teams = teams if teams else None
        logger.info(f"Loaded {len(teams)} known teams from corpus")
        return teams
    except Exception as e:
        logger.warning(f"Could not load known teams from corpus: {e}")
        return []


def _normalize_label(value: object) -> str:
    """Normalize a label to lowercase, stripped."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"

def run_triage_pipeline(
    title: str,
    description: str,
    labels: Optional[list[str]] = None,
    comments: Optional[str] = None,
) -> dict:
    """
    Execute the full ticket triage pipeline.

    Returns structured dict with classification, references,
    generated_response, and metadata.
    """
    start_time = time.time()
    stages_completed = []
    errors = []
    hyde_activated = False

    labels = labels or []
    comments = comments or ""

    result = {
        "classification": None,
        "retrieved_references": [],
        "generated_response": None,
        "metadata": {},
    }

    # ================================================================
    # Stage 1: Preprocessing
    # ================================================================
    try:
        logger.info("=" * 60)
        logger.info("Stage 1: Preprocessing ticket text")
        logger.info("=" * 60)

        issue_dict = {
            "title": title,
            "body": description,
            "labels": labels,
            "comments": [{"body": comments}] if comments else [],
        }
        processed = preprocess_issue(issue_dict)
        clean_title = processed["clean_title"]
        clean_body = processed["clean_body"]
        unified_text = processed["unified_text"]

        stages_completed.append("preprocessing")
        logger.info(f"Preprocessing complete. Unified text length: {len(unified_text)}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        errors.append(f"preprocessing: {str(e)}")
        clean_title = title
        clean_body = description
        unified_text = f"{title} . {description}"

    # ================================================================
    # Stage 2: Hybrid Retrieval
    # ================================================================
    retrieved_docs = []
    try:
        logger.info("=" * 60)
        logger.info("Stage 2: Hybrid retrieval (dense + BM25 + RRF)")
        logger.info("=" * 60)

        query_text = f"{clean_title} {clean_body}"
        retrieved_docs = hybrid_retrieve(query_text)

        stages_completed.append("retrieval")
        logger.info(f"Retrieved {len(retrieved_docs)} candidates")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        errors.append(f"retrieval: {str(e)}")

    # ================================================================
    # Stage 3: Reranking
    # ================================================================
    reranked_docs = []
    try:
        if retrieved_docs:
            logger.info("=" * 60)
            logger.info("Stage 3: Cross-encoder reranking (BAAI/bge-reranker-base)")
            logger.info("=" * 60)

            query_text = f"{clean_title} {clean_body}"
            reranked_docs = rerank(query_text, retrieved_docs)

            stages_completed.append("reranking")
            logger.info(f"Reranked to top {len(reranked_docs)} results")
        else:
            logger.warning("Skipping reranking - no retrieval results")

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        errors.append(f"reranking: {str(e)}")
        reranked_docs = retrieved_docs[:5]

    # ================================================================
    # Stage 4: Retrieval-Augmented Classification
    # ================================================================
    known_teams = _get_known_teams()
    try:
        logger.info("=" * 60)
        logger.info("Stage 4: Retrieval-Augmented Classification (Ollama llama3.1:8b)")
        logger.info("=" * 60)

        classification = classify_ticket(
            title=clean_title,
            description=clean_body,
            labels=labels,
            known_teams=known_teams if known_teams else None,
            retrieved_docs=reranked_docs[:10],  # top-10 for vote tally (Fix A+B)
        )
        result["classification"] = classification
        stages_completed.append("classification")

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        errors.append(f"classification: {str(e)}")
        result["classification"] = {
            "type": "bug",
            "severity": "medium",
            "team": "backend",
        }

    # ================================================================
    # Stage 4b: HyDE Fallback (classifier-retrieval agreement)
    # ================================================================
    if settings.HYDE_ENABLED and reranked_docs and result["classification"]:
        try:
            retrieved_team_list = [
                _normalize_label(
                    (doc.get("metadata", {}) if isinstance(doc, dict) else {}).get("team")
                )
                for doc in reranked_docs[:10]  # expand to 10 for consistent vote signal
            ]
            pred_team = _normalize_label(result["classification"].get("team"))
            matching = sum(1 for rt in retrieved_team_list if rt == pred_team)
            agreement = matching / max(len(retrieved_team_list), 1)

            logger.info(
                f"Classifier-retrieval agreement: {agreement:.2f} "
                f"(threshold: {settings.HYDE_AGREEMENT_THRESHOLD})"
            )

            if agreement < settings.HYDE_AGREEMENT_THRESHOLD:
                logger.info("=" * 60)
                logger.info("Stage 4b: HyDE activated (low classifier-retrieval agreement)")
                logger.info("=" * 60)

                query_text = f"{clean_title} {clean_body}"
                hyde_docs = hyde_retrieve(query_text)

                if hyde_docs:
                    hyde_reranked = rerank(query_text, hyde_docs)
                    hyde_confidence = get_top_confidence(hyde_reranked)
                    top_confidence = get_top_confidence(reranked_docs)

                    if hyde_confidence > top_confidence:
                        logger.info(
                            f"HyDE improved confidence: {top_confidence:.4f} → {hyde_confidence:.4f}"
                        )
                        reranked_docs = hyde_reranked

                        # Re-classify with improved retrieved docs
                        classification = classify_ticket(
                            title=clean_title,
                            description=clean_body,
                            labels=labels,
                            known_teams=known_teams if known_teams else None,
                            retrieved_docs=reranked_docs[:10],  # top-10 for vote tally (Fix A+B)
                        )
                        result["classification"] = classification
                        logger.info(f"HyDE re-classified team: {classification.get('team')}")
                    else:
                        logger.info("HyDE did not improve results, keeping original")

                hyde_activated = True
                stages_completed.append("hyde")
        except Exception as e:
            logger.error(f"HyDE fallback failed: {e}")
            errors.append(f"hyde: {str(e)}")

    # ================================================================
    # Stage 5: RAG Response Generation
    # ================================================================
    try:
        logger.info("=" * 60)
        logger.info("Stage 5: RAG response generation (Ollama llama3.1:8b)")
        logger.info("=" * 60)

        generated = generate_response(
            title=clean_title,
            description=clean_body,
            labels=labels,
            classification=result["classification"],
            retrieved_docs=reranked_docs,
        )
        result["generated_response"] = generated
        stages_completed.append("generation")

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        errors.append(f"generation: {str(e)}")
        result["generated_response"] = {
            "routing_explanation": "Unable to generate detailed response.",
            "debugging_steps": ["Review the issue manually"],
            "possible_causes": ["Automated analysis unavailable"],
        }

    # ================================================================
    # Build Retrieved References
    # ================================================================
    references = []
    for doc in reranked_docs:
        metadata = doc.get("metadata", {})
        references.append({
            "issue_number": metadata.get("issue_number", "N/A"),
            "title": metadata.get("title", "Unknown"),
            "similarity_score": round(
                doc.get("rerank_score", doc.get("rrf_score", 0)), 4
            ),
        })
    result["retrieved_references"] = references

    # ================================================================
    # Metadata
    # ================================================================
    elapsed_ms = int((time.time() - start_time) * 1000)
    result["metadata"] = {
        "processing_time_ms": elapsed_ms,
        "stages_completed": stages_completed,
        "errors": errors if errors else None,
        "num_retrieved": len(retrieved_docs),
        "num_reranked": len(reranked_docs),
        "hyde_activated": hyde_activated,
        "llm_backend": f"ollama/{settings.OLLAMA_MODEL}",
    }

    logger.info("=" * 60)
    logger.info(
        f"Pipeline complete in {elapsed_ms}ms. "
        f"Stages: {', '.join(stages_completed)}"
    )
    if hyde_activated:
        logger.info("HyDE was activated for this query")
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    logger.info("=" * 60)

    return result
