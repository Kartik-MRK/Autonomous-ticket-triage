"""
Pydantic Schema Models for the FastAPI Service
=================================================
Defines request and response models with validation
for the ticket triage API endpoints.

All models use Pydantic v2 for:
- Automatic validation
- JSON serialization
- OpenAPI schema generation (Swagger docs)
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================
# Request Models
# ============================================

class TicketInput(BaseModel):
    """Input model for a ticket to be triaged."""

    title: str = Field(
        ...,
        description="Title of the issue ticket",
        min_length=3,
        max_length=500,
        json_schema_extra={"example": "VSCode crashes when opening large file"},
    )

    description: str = Field(
        ...,
        description="Detailed description of the issue",
        min_length=10,
        json_schema_extra={
            "example": "The editor freezes and becomes unresponsive when opening files above 200MB. This happens consistently on Windows 11 with VSCode 1.85."
        },
    )

    labels: Optional[list[str]] = Field(
        default=[],
        description="Optional list of labels/tags for the ticket",
        json_schema_extra={"example": ["bug", "editor", "performance"]},
    )

    comments: Optional[str] = Field(
        default="",
        description="Optional additional comments or context",
        json_schema_extra={
            "example": "Multiple users reported this issue after the latest update."
        },
    )


# ============================================
# Response Models
# ============================================

class ClassificationResult(BaseModel):
    """Classification output for a ticket."""

    type: str = Field(
        ...,
        description="Issue type: bug, feature, or improvement",
        json_schema_extra={"example": "bug"},
    )

    severity: str = Field(
        ...,
        description="Severity level: low, medium, or high",
        json_schema_extra={"example": "high"},
    )

    team: str = Field(
        ...,
        description="Assigned team: frontend, backend, or infrastructure",
        json_schema_extra={"example": "backend"},
    )


class RetrievedReference(BaseModel):
    """A similar past issue retrieved from the knowledge base."""

    issue_number: str = Field(
        ...,
        description="GitHub issue number",
        json_schema_extra={"example": "12345"},
    )

    title: str = Field(
        ...,
        description="Title of the similar issue",
        json_schema_extra={"example": "Editor hangs on large JSON files"},
    )

    similarity_score: float = Field(
        ...,
        description="Relevance/similarity score",
        json_schema_extra={"example": 0.89},
    )


class GeneratedResponse(BaseModel):
    """AI-generated debugging suggestions and analysis."""

    routing_explanation: str = Field(
        ...,
        description="Explanation of why this ticket was routed to the assigned team",
        json_schema_extra={
            "example": "This issue relates to file handling and memory management in the core editor, making it a backend team responsibility."
        },
    )

    debugging_steps: list[str] = Field(
        ...,
        description="Ordered debugging steps",
        json_schema_extra={
            "example": [
                "1. Check file buffer allocation limits",
                "2. Inspect file loading module for memory leaks",
                "3. Review recent commits related to file handling",
            ]
        },
    )

    possible_causes: list[str] = Field(
        ...,
        description="Possible root causes",
        json_schema_extra={
            "example": [
                "Buffer overflow in file reader",
                "Missing file size validation",
            ]
        },
    )


class PipelineMetadata(BaseModel):
    """Metadata about the pipeline execution."""

    processing_time_ms: int = Field(
        ...,
        description="Total processing time in milliseconds",
    )

    stages_completed: list[str] = Field(
        ...,
        description="List of pipeline stages that completed successfully",
    )

    errors: Optional[list[str]] = Field(
        default=None,
        description="Any errors encountered during processing",
    )

    num_retrieved: int = Field(
        default=0,
        description="Number of documents retrieved",
    )

    num_reranked: int = Field(
        default=0,
        description="Number of documents after reranking",
    )


class TriageOutput(BaseModel):
    """Complete triage output including classification, references, and suggestions."""

    classification: ClassificationResult = Field(
        ...,
        description="Ticket classification results",
    )

    retrieved_references: list[RetrievedReference] = Field(
        default=[],
        description="Similar past issues retrieved from the knowledge base",
    )

    generated_response: GeneratedResponse = Field(
        ...,
        description="AI-generated debugging suggestions and analysis",
    )

    metadata: PipelineMetadata = Field(
        ...,
        description="Pipeline execution metadata",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        default="healthy",
        description="Service health status",
    )

    version: str = Field(
        default="1.0.0",
        description="API version",
    )

    vector_store_count: int = Field(
        default=0,
        description="Number of documents in the vector store",
    )
