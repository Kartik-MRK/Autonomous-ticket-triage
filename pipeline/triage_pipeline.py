"""
Triage Pipeline - End-to-End Orchestration
=============================================
Integrates all modules into a single sequential pipeline:
1. Preprocess ticket text (regex + spaCy)
2. Generate query embedding
3. Hybrid retrieval (dense + BM25)
4. Cross-encoder reranking
5. LLM classification (Gemini)
6. RAG response generation (Gemini)

Returns a final structured JSON containing classification results,
retrieved references, and generated suggestions.

Design Notes:
- Each stage is independent and can be tested/replaced separately
- Error handling at each stage with graceful degradation
- Pipeline returns partial results if downstream stages fail
- Supports both sync execution and async (for FastAPI)
"""

import time
from typing import Optional

from modules.preprocessing import preprocess_issue, clean_text_regex
from modules.retrieval import hybrid_retrieve
from modules.reranker import rerank
from modules.classifier import classify_ticket
from modules.generator import generate_response
from utils.logger import get_logger

logger = get_logger(__name__)


def run_triage_pipeline(
    title: str,
    description: str,
    labels: Optional[list[str]] = None,
    comments: Optional[str] = None,
) -> dict:
    """
    Execute the full ticket triage pipeline.

    Sequentially performs:
    1. Text preprocessing
    2. Hybrid retrieval (dense + BM25)
    3. Cross-encoder reranking
    4. Ticket classification via Gemini
    5. RAG-based response generation via Gemini

    Args:
        title: Ticket title.
        description: Ticket description/body.
        labels: Optional list of label strings.
        comments: Optional comments text.

    Returns:
        Structured dictionary with:
        - classification: {type, severity, team}
        - retrieved_references: [{issue_number, title, similarity_score}]
        - generated_response: {routing_explanation, debugging_steps, possible_causes}
        - metadata: {processing_time_ms, stages_completed}
    """
    start_time = time.time()
    stages_completed = []
    errors = []

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

        # Build a mock issue dict for the preprocessor
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
        # Fallback: use raw text
        clean_title = title
        clean_body = description
        unified_text = f"{title} . {description}"

    # ================================================================
    # Stage 2: Hybrid Retrieval
    # ================================================================
    retrieved_docs = []
    try:
        logger.info("=" * 60)
        logger.info("Stage 2: Hybrid retrieval (dense + BM25)")
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
            logger.info("Stage 3: Cross-encoder reranking")
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
        reranked_docs = retrieved_docs[:5]  # Fallback to un-reranked results

    # ================================================================
    # Stage 4: Classification
    # ================================================================
    try:
        logger.info("=" * 60)
        logger.info("Stage 4: Ticket classification (Gemini)")
        logger.info("=" * 60)

        classification = classify_ticket(
            title=clean_title,
            description=clean_body,
            labels=labels,
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
    # Stage 5: RAG Response Generation
    # ================================================================
    try:
        logger.info("=" * 60)
        logger.info("Stage 5: RAG response generation (Gemini)")
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
    }

    logger.info("=" * 60)
    logger.info(
        f"Pipeline complete in {elapsed_ms}ms. "
        f"Stages: {', '.join(stages_completed)}"
    )
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    logger.info("=" * 60)

    return result
