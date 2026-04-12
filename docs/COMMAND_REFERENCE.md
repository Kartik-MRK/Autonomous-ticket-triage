# Command Reference

## Core Setup

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Data and Index

python scripts/ingest_bugzilla_core.py --target-bugs 800
python scripts/build_bugzilla_index.py --rebuild-clean --reset
python scripts/preprocess_bugzilla_raw.py

## Query

python scripts/run_bugzilla_pipeline.py
python main.py run-cli
python main.py pipeline --title "sample" --description "sample text"

## UI

python -m streamlit run streamlit_app.py
python main.py ui
python main.py ui --port 8601

## API

python main.py serve
python scripts/test_endpoint.py --url http://localhost:8000

## Diagnostics

python -m streamlit --version
python main.py --help
python scripts/build_bugzilla_index.py --help
python scripts/ingest_bugzilla_core.py --help
