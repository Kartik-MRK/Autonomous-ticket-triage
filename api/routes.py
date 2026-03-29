"""
API Routes Module
===================
Defines the FastAPI route handlers for the ticket triage service.

Endpoints:
- POST /triage — Submit a ticket for automated triage
- GET  /health — Service health check
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    TicketInput,
    TriageOutput,
    ClassificationResult,
    RetrievedReference,
    GeneratedResponse,
    PipelineMetadata,
    HealthResponse,
)
from pipeline.triage_pipeline import run_triage_pipeline
from modules.vector_store import get_collection_stats
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/triage",
    response_model=TriageOutput,
    summary="Triage a ticket",
    description=(
        "Submit a software issue ticket for automated triage. "
        "The system will classify the ticket, retrieve similar past issues, "
        "and generate debugging suggestions using RAG."
    ),
    tags=["Triage"],
)
async def triage_ticket(ticket: TicketInput):
    """
    Process a ticket through the full triage pipeline.

    The pipeline performs:
    1. Text preprocessing (regex + spaCy)
    2. Hybrid retrieval (dense + BM25)
    3. Cross-encoder reranking
    4. Classification via Gemini
    5. RAG-based response generation
    """
    try:
        logger.info(f"Received triage request: '{ticket.title[:60]}...'")

        # Run the full pipeline
        result = run_triage_pipeline(
            title=ticket.title,
            description=ticket.description,
            labels=ticket.labels,
            comments=ticket.comments,
        )

        # Convert to response model
        classification = ClassificationResult(**result["classification"])

        references = [
            RetrievedReference(
                issue_number=str(ref.get("issue_number", "N/A")),
                title=ref.get("title", "Unknown"),
                similarity_score=ref.get("similarity_score", 0.0),
            )
            for ref in result.get("retrieved_references", [])
        ]

        generated = GeneratedResponse(**result["generated_response"])

        metadata = PipelineMetadata(**result["metadata"])

        response = TriageOutput(
            classification=classification,
            retrieved_references=references,
            generated_response=generated,
            metadata=metadata,
        )

        logger.info(
            f"Triage complete: type={classification.type}, "
            f"severity={classification.severity}, team={classification.team}"
        )

        return response

    except Exception as e:
        logger.error(f"Triage endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and get basic stats.",
    tags=["System"],
)
async def health_check():
    """Return service health status and vector store statistics."""
    try:
        stats = get_collection_stats()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_count=stats.get("count", 0),
        )
    except Exception as e:
        logger.warning(f"Health check partial failure: {e}")
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            vector_store_count=0,
        )
