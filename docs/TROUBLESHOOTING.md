# Troubleshooting

## UI Does Not Start

### Symptom
You run a command but UI does not open.

### Fix
Use exact command:

python -m streamlit run streamlit_app.py

Do not use:
- streamlit_app.py
- streamlit streamlit_app.py

## Streamlit Command Not Found

### Symptom
streamlit is not recognized.

### Fix
1. Activate venv:

.\.venv\Scripts\Activate.ps1

2. Reinstall dependencies:

pip install -r requirements.txt

3. Run with Python module style:

python -m streamlit run streamlit_app.py

## No Indexed Data

### Symptom
UI shows warning that index is empty.

### Fix
Run:

python scripts/ingest_bugzilla_core.py --target-bugs 800
python scripts/build_bugzilla_index.py --rebuild-clean --reset

## spaCy Model Error

### Symptom
en_core_web_sm not found.

### Fix
python -m spacy download en_core_web_sm

## Gemini Errors (404 or 429)

### 404 model not found
Set .env model to a valid name.
Current default in project: gemini-2.0-flash.

### 429 quota exhausted
Your API quota is exhausted.
Retrieval pipeline still runs; classification/generation falls back.

## Slow First Query

### Why
Models are downloaded or loaded into memory on first run.

### What to do
Wait for first run to complete.
Second run is much faster.

## Port Already In Use

### Symptom
Streamlit fails on 8501.

### Fix
Use another port:

python main.py ui --port 8601

or

python -m streamlit run streamlit_app.py --server.port 8601
