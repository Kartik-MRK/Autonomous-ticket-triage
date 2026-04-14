"""
FastAPI Server Module
=======================
Initializes and configures the FastAPI application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("Starting Autonomous Ticket Triage Service (Ollama)")
    logger.info("=" * 60)
    warnings = settings.validate()
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")
    settings.ensure_directories()
    logger.info(f"LLM backend: Ollama/{settings.OLLAMA_MODEL}")
    logger.info(f"Swagger UI: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down Autonomous Ticket Triage Service")


app = FastAPI(
    title="Autonomous Ticket Triage and Routing API",
    description=(
        "AI-powered ticket triage using RAG and local LLM (Ollama).\n\n"
        "**Features:**\n"
        "- 🔍 Hybrid retrieval (dense + BM25 + RRF)\n"
        "- 🎯 Cross-encoder reranking (BAAI/bge-reranker-base)\n"
        "- 🧪 HyDE fallback for low-confidence queries\n"
        "- 🏷️ Automated classification via Ollama\n"
        "- 💡 RAG-based debugging suggestions\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router)
