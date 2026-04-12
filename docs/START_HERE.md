# Start Here

This guide gives you the shortest path from clone to working UI.

## 1. Open Project Folder

Open terminal in project root:

E:\College_Documents\Sem-6\GenAI Project\Autonomous ticket triage

## 2. Create and Activate Virtual Environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

## 3. Install Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

## 4. Configure Environment

Copy .env.example to .env and set your key:

GOOGLE_API_KEY=your_key_here

Notes:
- If key is missing or quota is exhausted, retrieval still works.
- Classification and generation stages fall back to default responses.

## 5. Build Data and Index

Step 1 fetch raw Bugzilla Core records:

python scripts/ingest_bugzilla_core.py --target-bugs 800

Step 2 preprocess and build Chroma index:

python scripts/build_bugzilla_index.py --rebuild-clean --reset

## 6. Start UI

Use one of the following:

python -m streamlit run streamlit_app.py

or

python main.py ui

Open browser:

http://localhost:8501

## 7. First Query

In the UI:
- Enter Ticket title
- Enter Ticket description
- Click Run RAG Triage

You will get:
- Classification
- Routing explanation
- Debugging steps
- Possible causes
- Retrieved Bugzilla references
