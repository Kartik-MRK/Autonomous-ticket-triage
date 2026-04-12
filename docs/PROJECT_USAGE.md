# Project Usage

This document explains how to use the full Mozilla Core triage workflow.

## Workflow Overview

1. Ingest raw Bugzilla Core issues
2. Build clean dataset and vector index
3. Query via CLI, API, or Streamlit UI

## A. Data Ingestion

python scripts/ingest_bugzilla_core.py --target-bugs 800

Output:
- data/raw/bugzilla_core_raw_issues.json

## B. Preprocess + Index

python scripts/build_bugzilla_index.py --rebuild-clean --reset

This does:
- Clean record generation from raw data
- Regex + spaCy preprocessing
- Embedding generation
- ChromaDB upsert

Outputs:
- data/processed/bugzilla_core_clean_dataset.json
- data/processed/bugzilla_core_spacy_processed.json
- data/chroma_db/chroma.sqlite3

## C. Query Modes

### 1) Interactive CLI

python scripts/run_bugzilla_pipeline.py

Type query and press Enter.
Type exit to quit.

### 2) Streamlit UI

python -m streamlit run streamlit_app.py

### 3) FastAPI

Start API:

python main.py serve

Open docs:

http://localhost:8000/docs

Test endpoint:

python scripts/test_endpoint.py

## D. Main Command Shortcuts

python main.py ingest-bugzilla --target-bugs 800
python main.py build-bugzilla-index --rebuild-clean --reset
python main.py run-cli
python main.py ui
python main.py pipeline --title "..." --description "..."
python main.py serve
