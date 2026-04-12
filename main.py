"""
Mozilla Core Autonomous Ticket Triage
====================================
Main CLI entry point focused on Bugzilla Core data.

Core workflow:
1) python main.py ingest-bugzilla
2) python main.py build-bugzilla-index --rebuild-clean --reset
3) python main.py run-cli

Optional frontend:
- python main.py ui
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def cmd_serve(args) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from utils.config import settings

    print("Starting Mozilla Core Triage API...")
    print(f"Swagger UI: http://localhost:{settings.API_PORT}/docs")

    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=args.reload,
        log_level="info",
    )


def cmd_ingest_bugzilla(args) -> None:
    """Fetch raw Mozilla Bugzilla Core bugs and save under data/raw."""
    from modules.bugzilla_ingestion import (
        build_bugzilla_core_raw_data,
        save_bugzilla_raw_data,
    )

    raw_data = build_bugzilla_core_raw_data(
        target_count=args.target_bugs,
        page_size=args.page_size,
        bug_list_delay=args.list_delay,
        comment_delay=args.comment_delay,
    )

    if not raw_data:
        print("No Bugzilla records were produced. Check logs and try again.")
        sys.exit(1)

    output_path = save_bugzilla_raw_data(raw_data, output_path=args.output)
    print(f"Done! {len(raw_data)} raw Bugzilla records saved to {output_path}")


def cmd_build_bugzilla_index(args) -> None:
    """Build clean Bugzilla data, run regex+spaCy, and index in ChromaDB."""
    from modules.bugzilla_index_builder import build_bugzilla_index

    summary = build_bugzilla_index(
        raw_input=args.raw_input,
        clean_input=args.clean_input,
        processed_output=args.processed_output,
        max_records=args.max_records,
        rebuild_clean=args.rebuild_clean,
        reset=args.reset,
    )

    stats = summary.get("collection_stats", {})
    print(
        "Done! "
        f"clean={summary.get('clean_count', 0)}, "
        f"spacy_processed={summary.get('spacy_processed_count', 0)}, "
        f"chroma_count={stats.get('count', 0)}"
    )


def cmd_pipeline(args) -> None:
    """Run one query through the full triage pipeline."""
    from pipeline.triage_pipeline import run_triage_pipeline

    title = args.title or "Firefox crashes while opening a large media stream"
    description = args.description or (
        "Firefox becomes unresponsive and eventually crashes when a large media "
        "stream starts in a WebRTC session on Linux."
    )
    labels = args.labels.split(",") if args.labels else ["mozilla-core", "bug"]

    result = run_triage_pipeline(
        title=title,
        description=description,
        labels=[label.strip() for label in labels if label.strip()],
        comments=args.comments or "",
    )

    print("\n" + "=" * 60)
    print("TRIAGE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))


def cmd_run_cli(_args) -> None:
    """Start interactive CLI query loop."""
    subprocess.run([sys.executable, "scripts/run_bugzilla_pipeline.py"], check=False)


def cmd_ui(args) -> None:
    """Launch the Streamlit frontend."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        str(args.port),
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mozilla Core Autonomous Ticket Triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  python main.py ingest-bugzilla --target-bugs 800\n"
            "  python main.py build-bugzilla-index --rebuild-clean --reset\n"
            "  python main.py run-cli\n"
            "  python main.py ui"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Serve API ----
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # ---- Ingest Bugzilla ----
    ingest_bugzilla_parser = subparsers.add_parser(
        "ingest-bugzilla",
        help="Fetch raw Mozilla Bugzilla Core bugs",
    )
    ingest_bugzilla_parser.add_argument(
        "--target-bugs",
        type=int,
        default=800,
        help="Target number of Bugzilla bugs to fetch",
    )
    ingest_bugzilla_parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Page size for Bugzilla pagination",
    )
    ingest_bugzilla_parser.add_argument(
        "--list-delay",
        type=float,
        default=0.25,
        help="Delay in seconds between bug list API calls",
    )
    ingest_bugzilla_parser.add_argument(
        "--comment-delay",
        type=float,
        default=0.1,
        help="Delay in seconds between comment API calls",
    )
    ingest_bugzilla_parser.add_argument(
        "--output",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Raw output JSON path",
    )

    # ---- Build Bugzilla Index ----
    build_bugzilla_parser = subparsers.add_parser(
        "build-bugzilla-index",
        help="Build clean dataset, regex+spaCy data, and Chroma index",
    )
    build_bugzilla_parser.add_argument(
        "--raw-input",
        type=str,
        default="data/raw/bugzilla_core_raw_issues.json",
        help="Path to raw Bugzilla dataset JSON",
    )
    build_bugzilla_parser.add_argument(
        "--clean-input",
        type=str,
        default="data/processed/bugzilla_core_clean_dataset.json",
        help="Path to clean Bugzilla dataset JSON",
    )
    build_bugzilla_parser.add_argument(
        "--processed-output",
        type=str,
        default="data/processed/bugzilla_core_spacy_processed.json",
        help="Output for regex+spaCy processed records",
    )
    build_bugzilla_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for clean dataset build",
    )
    build_bugzilla_parser.add_argument(
        "--rebuild-clean",
        action="store_true",
        help="Force rebuilding clean dataset from raw",
    )
    build_bugzilla_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset Chroma collection before indexing",
    )

    # ---- One-off pipeline ----
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run a single query through the triage pipeline",
    )
    pipeline_parser.add_argument("--title", type=str, help="Ticket title")
    pipeline_parser.add_argument("--description", type=str, help="Ticket description")
    pipeline_parser.add_argument("--labels", type=str, help="Comma-separated labels")
    pipeline_parser.add_argument("--comments", type=str, help="Additional comments")

    # ---- Interactive CLI ----
    subparsers.add_parser(
        "run-cli",
        help="Open interactive terminal query loop",
    )

    # ---- Streamlit UI ----
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch Streamlit frontend",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "serve": cmd_serve,
        "ingest-bugzilla": cmd_ingest_bugzilla,
        "build-bugzilla-index": cmd_build_bugzilla_index,
        "pipeline": cmd_pipeline,
        "run-cli": cmd_run_cli,
        "ui": cmd_ui,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
