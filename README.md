# Mozilla Core Autonomous Ticket Triage

Mozilla Bugzilla Core focused RAG pipeline.

Project flow:
1. Fetch raw Bugzilla Core data
2. Preprocess with regex + spaCy
3. Build ChromaDB vector store
4. Run triage from CLI, API, or Streamlit UI

## Quick Run (Most Important)

From project root, run in order:

1) python scripts/ingest_bugzilla_core.py --target-bugs 800
2) python scripts/build_bugzilla_index.py --rebuild-clean --reset
3) python scripts/run_bugzilla_pipeline.py

To open UI:

python -m streamlit run streamlit_app.py

Alternative UI shortcut:

python main.py ui

## UI Launch Note

Correct Streamlit command must include run:

python -m streamlit run streamlit_app.py

If you run streamlit streamlit_app.py, UI may not start.

## Documentation Index

- [Start here](docs/START_HERE.md)
- [Run UI guide](docs/RUN_UI.md)
- [Project usage](docs/PROJECT_USAGE.md)
- [Command reference](docs/COMMAND_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## API

Start API:

python main.py serve

API docs:
- http://localhost:8000/docs
- http://localhost:8000/health

## Core Files

- main.py
- streamlit_app.py
- scripts/ingest_bugzilla_core.py
- scripts/build_bugzilla_index.py
- scripts/run_bugzilla_pipeline.py
- modules/bugzilla_ingestion.py
- modules/bugzilla_index_builder.py
