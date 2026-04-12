# Run UI Guide

This file explains exactly how to run Streamlit UI for this project.

## Correct Command

The correct Streamlit command is:

python -m streamlit run streamlit_app.py

Important:
- Do not run: streamlit_app.py
- Do not run: streamlit streamlit_app.py
- The run keyword is required.

## Full Windows Steps

1. Open PowerShell in project root.
2. Activate virtual environment:

.\.venv\Scripts\Activate.ps1

3. Start UI:

python -m streamlit run streamlit_app.py

4. Open shown local URL (usually http://localhost:8501).

## Alternate Command Through Main CLI

python main.py ui

Custom port:

python main.py ui --port 8601

## Verify Streamlit Installation

python -m streamlit --version

If this fails:

pip install -r requirements.txt

## Stop UI

Press Ctrl + C in the terminal where Streamlit is running.
