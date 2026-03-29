"""
FastAPI Server Module
=======================
Initializes and configures the FastAPI application with:
- CORS middleware for cross-origin requests
- Lifespan handler for startup/shutdown
- Route registration
- Swagger/OpenAPI documentation

Start with: uvicorn api.server:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs initialization on startup and cleanup on shutdown.
    """
    # ---- Startup ----
    logger.info("=" * 60)
    logger.info("Starting Autonomous Ticket Triage Service")
    logger.info("=" * 60)

    # Validate configuration
    warnings = settings.validate()
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")

    # Ensure data directories exist
    settings.ensure_directories()

    # Pre-load models (optional - can be deferred to first request)
    logger.info("Service ready. Models will be loaded on first request.")
    logger.info(f"Swagger UI available at http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("=" * 60)

    yield  # Application runs here

    # ---- Shutdown ----
    logger.info("Shutting down Autonomous Ticket Triage Service")


# ============================================
# Create FastAPI Application
# ============================================
app = FastAPI(
    title="Autonomous Ticket Triage and Routing API",
    description=(
        "An AI-powered service that automatically triages software issue tickets "
        "using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). "
        "\n\n"
        "**Features:**\n"
        "- 🔍 Hybrid retrieval (dense embeddings + BM25)\n"
        "- 🎯 Cross-encoder reranking for precision\n"
        "- 🏷️ Automated classification (type, severity, team)\n"
        "- 💡 RAG-based debugging suggestions\n"
        "- 📊 Grounded in historical issue data\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================
# CORS Middleware
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Register Routes
# ============================================
app.include_router(router, prefix="/api/v1")

# Also register at root for convenience
app.include_router(router)
