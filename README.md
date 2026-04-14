# 🛰️ Mozilla Core — Autonomous Ticket Triage

An AI-powered system that automatically triages Mozilla Bugzilla Core bug reports using **Retrieval-Augmented Generation (RAG)** with a local LLM (Ollama llama3.1:8b). The pipeline classifies tickets, routes them to the correct team, and generates debugging suggestions grounded in historical issue data.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                 │
│                (bug title + description)                          │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                     ┌─────▼─────┐
                     │  Stage 1  │  Regex + spaCy Preprocessing
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │  Stage 2  │  Hybrid Retrieval
                     │           │  (Dense/ChromaDB + BM25 → RRF Fusion)
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │  Stage 3  │  Cross-Encoder Reranking
                     │           │  (BAAI/bge-reranker-base)
                     └─────┬─────┘
                           │
                    ┌──────▼──────┐     Low confidence?
                    │  Stage 3b   │◄─── Yes ──► HyDE Fallback
                    │  (HyDE)     │     Generate hypothetical doc,
                    └──────┬──────┘     re-retrieve, re-rerank
                           │
                     ┌─────▼─────┐
                     │  Stage 4  │  Classification
                     │           │  (Ollama llama3.1:8b)
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │  Stage 5  │  RAG Response Generation
                     │           │  (Ollama llama3.1:8b)
                     └─────┬─────┘
                           │
              ┌────────────▼────────────┐
              │   STRUCTURED OUTPUT     │
              │ • Classification        │
              │ • Routing explanation   │
              │ • Debugging steps       │
              │ • Possible causes       │
              │ • Similar issue refs    │
              └─────────────────────────┘
```

## Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Preprocessing** | Regex + spaCy | Clean text, tokenize, extract entities |
| **Embeddings** | BAAI/bge-large-en | 1024-dim dense vectors for semantic search |
| **Vector Store** | ChromaDB | Persistent vector database with cosine similarity |
| **Sparse Search** | BM25 (rank-bm25) | Keyword-based retrieval for lexical matching |
| **Fusion** | Reciprocal Rank Fusion | Merge dense + sparse results (k=60) |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder reranking for precision |
| **HyDE** | Ollama llama3.1:8b | Hypothetical document generation for low-confidence queries |
| **Classification** | Ollama llama3.1:8b | Type/severity/team classification |
| **Generation** | Ollama llama3.1:8b | RAG-grounded debugging suggestions |

## Folder Structure

```
Autonomous ticket triage/
├── .env                            # Configuration (gitignored)
├── .env.example                    # Config template
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                         # Single CLI entry point
├── streamlit_app.py                # Streamlit UI
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Centralized configuration
│
├── modules/
│   ├── __init__.py
│   ├── ingestion.py                # Bugzilla data fetching
│   ├── preprocessing.py            # Regex + spaCy pipeline
│   ├── embedding.py                # BAAI/bge-large-en embeddings
│   ├── vector_store.py             # ChromaDB operations
│   ├── retrieval.py                # Hybrid retrieval (dense + BM25 + RRF)
│   ├── reranker.py                 # BAAI/bge-reranker-base cross-encoder
│   ├── hyde.py                     # HyDE — Hypothetical Document Embeddings
│   ├── classifier.py               # Ticket classification (Ollama)
│   └── generator.py                # RAG response generation (Ollama)
│
├── pipeline/
│   ├── __init__.py
│   ├── index_builder.py            # Build ChromaDB index + test split
│   └── triage_pipeline.py          # End-to-end orchestration
│
├── api/
│   ├── __init__.py
│   ├── server.py                   # FastAPI application
│   ├── routes.py                   # API endpoints
│   └── schemas.py                  # Pydantic models
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                 # Evaluation runner
│   └── metrics.py                  # Retrieval + classification metrics
│
├── tests/
│   ├── .env                        # Test-specific Ollama config
│   ├── ollama_helpers.py           # Shared Ollama helpers
│   ├── eval_engine.py              # Shared evaluation engine
│   ├── dense_only_retrival/
│   │   ├── run_eval.py             # Dense-only evaluation
│   │   └── results/                # evaluation_metrics.json, query_responses.json
│   ├── sparse_only_retrival/
│   │   ├── run_eval.py             # Sparse-only evaluation
│   │   └── results/
│   └── hybrid_retrival/
│       ├── run_eval.py             # Hybrid evaluation
│       └── results/
│
├── utils/
│   ├── __init__.py
│   └── logger.py                   # Structured logging
│
├── data/                           # Generated data (gitignored)
│   ├── raw/
│   │   └── bugzilla_core_raw.json
│   ├── processed/
│   │   ├── bugzilla_core_clean.json
│   │   ├── bugzilla_core_processed.json
│   │   └── test_processed.json     # 100-issue test split
│   └── chroma_db/
│
└── logs/
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally with `llama3.1:8b`:
   ```bash
   ollama pull llama3.1:8b
   ollama serve
   ```
3. **spaCy English model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Step-by-Step Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd "Autonomous ticket triage"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
copy .env.example .env
```

Edit `.env` if needed (defaults work out of the box with Ollama running locally).

### 3. Verify Ollama is running

```bash
ollama list
```

You should see `llama3.1:8b` in the list. If not:
```bash
ollama pull llama3.1:8b
```

### 4. Fetch raw Bugzilla data (3000 issues)

```bash
python main.py ingest
```

This fetches 3000 Mozilla Core bugs with their comments from the Bugzilla REST API.
Output: `data/raw/bugzilla_core_raw.json`

### 5. Build the index

```bash
python main.py build-index
```

This command:
- Cleans the raw data (removes issues without solutions)
- Randomly splits 100 issues into `data/processed/test_processed.json`
- Runs regex + spaCy preprocessing on the remaining issues
- Generates BAAI/bge-large-en embeddings
- Stores everything in ChromaDB

### 6. Query the system (Interactive CLI)

```bash
python main.py query
```

Type any bug description and get:
- Classification (type, severity, team)
- Debugging steps
- Possible root causes
- Similar historical issues

### 7. Launch the Streamlit UI

```bash
python main.py ui
```

Or directly:
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 8. Run evaluations

Each retrieval mode has its own command:

```bash
python main.py eval-dense     # Dense-only retrieval (no HyDE)
python main.py eval-sparse    # Sparse-only / BM25 retrieval (no HyDE)
python main.py eval-hybrid    # Hybrid retrieval with HyDE fallback
```

Results are saved to each test's `results/` folder:
- `tests/dense_only_retrival/results/` — dense evaluation
- `tests/sparse_only_retrival/results/` — sparse evaluation
- `tests/hybrid_retrival/results/` — hybrid evaluation

Each produces:
- `evaluation_metrics.json` — aggregate metrics (precision, recall, F1, MRR, nDCG)
- `query_responses.json` — per-query results with classification, references, generated response

> **Note:** Only `eval-hybrid` uses HyDE. HyDE fires at most **once** per query when reranker confidence is below the threshold.

## HyDE — Hypothetical Document Embeddings

When the reranker confidence score falls below a threshold (default: 0.3), the system automatically activates **HyDE**:

1. Uses Ollama to generate a hypothetical "ideal resolved bug report" from the query
2. Embeds this hypothetical document (without the query instruction prefix)
3. Re-runs dense + sparse retrieval using the hypothetical embedding/text
4. Re-ranks the new results
5. If the new results have higher confidence, they replace the original results

This fires **once only** — no loops or repeated attempts.

This technique is based on [Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)](https://arxiv.org/abs/2212.10496).

Configure in `.env`:
```
HYDE_ENABLED=true
HYDE_CONFIDENCE_THRESHOLD=0.3
```

## API

Start the FastAPI server:
```bash
python main.py serve
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Triage endpoint: `POST /triage`

## All Commands Summary

| Command | Description |
|---------|-------------|
| `python main.py ingest` | Fetch 3000 raw Bugzilla issues |
| `python main.py build-index` | Preprocess, split test set, build ChromaDB |
| `python main.py query` | Interactive CLI triage |
| `python main.py ui` | Launch Streamlit UI |
| `python main.py serve` | Start FastAPI server |
| `python main.py eval-dense` | Evaluate with dense-only retrieval (no HyDE) |
| `python main.py eval-sparse` | Evaluate with sparse-only retrieval (no HyDE) |
| `python main.py eval-hybrid` | Evaluate with hybrid retrieval + HyDE |

**All commands work with zero arguments** — no explicit values needed.
