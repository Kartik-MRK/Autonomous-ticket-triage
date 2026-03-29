# Autonomous Ticket Triage and Routing Using RAG and LLMs

An AI-powered system that automatically triages software issue tickets using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). The system classifies issues, routes them to the appropriate team, and generates debugging suggestions grounded in historical issue data.

## 🏗️ Architecture Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GitHub API  │───>│ Preprocessing│───>│  Embedding   │
│  (GraphQL)   │    │ (Regex+spaCy)│    │ (BGE-large)  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │   ChromaDB   │
                                        │ Vector Store │
                                        └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│   Response   │<───│  Classifier  │<───│   Hybrid     │
│  Generation  │    │  (Gemini)    │    │  Retrieval   │
│  (RAG+LLM)   │    └──────────────┘    │ Dense + BM25 │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       │                                ┌──────▼───────-┐
       │                                │  Reranking    │
       │                                │(Cross-Encoder)│
       │                                └──────────────-┘
       ▼
┌──────────────┐
│  FastAPI     │
│  Endpoint    │
└──────────────┘
```

## 📋 Features

- **Data Ingestion**: Fetch GitHub issues via GraphQL API with pagination and filtering
- **Hybrid Preprocessing**: Regex cleaning + spaCy NLP (tokenization, lemmatization, NER)
- **Dense Embeddings**: BAAI/bge-large-en model with instruction prefixing
- **Persistent Vector Store**: ChromaDB with metadata storage
- **Hybrid Retrieval**: Dense (cosine similarity) + Sparse (BM25) with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-12-v2 for precise relevance scoring
- **LLM Classification**: Gemini-based structured classification (type, severity, team)
- **RAG Generation**: Context-grounded debugging suggestions using retrieved similar issues
- **FastAPI Service**: REST API with Swagger documentation
- **Evaluation**: Classification metrics (F1, accuracy) and retrieval metrics (Hit@K, Recall@K)

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- GitHub Personal Access Token (with `repo` scope)
- Google AI Studio API Key

### 1. Clone and Setup Virtual Environment

```bash
cd "Autonomous ticket triage"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GITHUB_TOKEN=your_github_personal_access_token
GOOGLE_API_KEY=your_google_ai_studio_api_key
```

### 5. Ingest Data from GitHub

```bash
python main.py ingest --max-issues 500
```

This fetches 500 issues from the Microsoft VSCode repository using the GraphQL API.

### 6. Preprocess Data

```bash
python main.py preprocess
```

### 7. Build Vector Store Index

```bash
python main.py build-index --reset
```

This generates embeddings and stores them in ChromaDB. First run downloads the embedding model (~1.2GB).

### 8. Start the API Server

```bash
python main.py serve
```

The server starts at `http://localhost:8000`. Access Swagger UI at `http://localhost:8000/docs`.

### 9. Test the Endpoint

In a new terminal:

```bash
# Using the test script:
python scripts/test_endpoint.py

# Or using curl:
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "title": "VSCode crashes when opening large file",
    "description": "The editor freezes when opening files above 200MB on Windows 11.",
    "labels": ["bug", "editor"],
    "comments": "Started after the latest update."
  }'
```

### 10. Run Evaluation

```bash
python main.py evaluate --max-samples 50
```

## 📁 Project Structure

```
├── .env                    # API keys and configuration
├── .env.example            # Template for .env
├── requirements.txt        # Python dependencies
├── main.py                 # CLI entry point
├── README.md               # This file
│
├── api/                    # FastAPI service layer
│   ├── server.py           # App initialization, CORS, lifespan
│   ├── routes.py           # API endpoints (/triage, /health)
│   └── schemas.py          # Pydantic request/response models
│
├── pipeline/               # End-to-end orchestration
│   └── triage_pipeline.py  # Sequential pipeline function
│
├── modules/                # Core ML/AI modules
│   ├── ingestion.py        # GitHub GraphQL data fetcher
│   ├── preprocessing.py    # Regex + spaCy text cleaning
│   ├── embedding.py        # BAAI/bge-large-en embeddings
│   ├── vector_store.py     # ChromaDB operations
│   ├── retrieval.py        # Hybrid dense + BM25 retrieval
│   ├── reranker.py         # Cross-encoder reranking
│   ├── classifier.py       # Gemini classification
│   └── generator.py        # Gemini RAG generation
│
├── evaluation/             # Evaluation & metrics
│   ├── metrics.py          # Accuracy, F1, Hit@K, Recall@K
│   └── evaluate.py         # Evaluation runner
│
├── data/                   # Data storage (auto-created)
│   ├── raw/                # Raw fetched issues
│   ├── processed/          # Cleaned/preprocessed data
│   └── chroma_db/          # ChromaDB persistent storage
│
├── utils/                  # Shared utilities
│   ├── config.py           # Environment & settings
│   └── logger.py           # Structured logging
│
└── scripts/                # Helper scripts
    ├── ingest_data.py      # Standalone ingestion
    ├── build_index.py      # Standalone index building
    └── test_endpoint.py    # API endpoint tester
```

## 🔧 Configuration

All configuration is managed through the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | - | GitHub PAT for GraphQL API |
| `GOOGLE_API_KEY` | - | Google AI Studio key for Gemini |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en` | Sentence transformer model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Cross-encoder model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `RETRIEVAL_TOP_K` | `20` | Dense retrieval candidates |
| `BM25_TOP_K` | `20` | BM25 retrieval candidates |
| `RERANK_TOP_N` | `5` | Final reranked results |
| `MAX_ISSUES` | `500` | Issues to fetch from GitHub |
| `API_PORT` | `8000` | FastAPI server port |

## 📊 API Endpoints

### POST /triage

Submit a ticket for automated triage.

**Request:**
```json
{
  "title": "VSCode crashes when opening large file",
  "description": "The editor freezes when opening files above 200MB.",
  "labels": ["bug", "editor"],
  "comments": "Started after the latest update."
}
```

**Response:**
```json
{
  "classification": {
    "type": "bug",
    "severity": "high",
    "team": "backend"
  },
  "retrieved_references": [
    {
      "issue_number": "12345",
      "title": "Editor hangs on large JSON files",
      "similarity_score": 0.89
    }
  ],
  "generated_response": {
    "routing_explanation": "This issue relates to file handling...",
    "debugging_steps": ["1. Check file buffer allocation", "..."],
    "possible_causes": ["Buffer overflow", "Missing validation"]
  },
  "metadata": {
    "processing_time_ms": 3500,
    "stages_completed": ["preprocessing", "retrieval", "reranking", "classification", "generation"]
  }
}
```

### GET /health

Health check endpoint.

## 🔮 Future Enhancements

- **Caching**: Redis-based caching for repeated queries
- **Batch processing**: Process multiple tickets in parallel
- **Advanced reranking**: ColBERT or additional reranking strategies
- **Fine-tuning**: Fine-tune embedding model on domain-specific data
- **Feedback loop**: Learn from user corrections to improve accuracy
- **Multi-language support**: Handle non-English tickets
- **Dashboard**: Web UI for real-time triage monitoring

## 📝 License

This project is for educational purposes as part of the Gen AI course (Semester 6).
