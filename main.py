"""
Mozilla Core Autonomous Ticket Triage
======================================
Main CLI entry point. All commands work with zero arguments.

Usage:
    python main.py ingest            # Fetch 3000 raw Bugzilla issues
    python main.py build-index       # Preprocess, split test set, build ChromaDB
    python main.py query             # Interactive CLI triage
    python main.py ui                # Launch Streamlit UI
    python main.py serve             # Start FastAPI server
    python main.py evaluate          # Run evaluation on test set
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def cmd_ingest(args) -> None:
    """Fetch raw Mozilla Bugzilla Core bugs."""
    from modules.ingestion import build_bugzilla_core_raw_data, save_bugzilla_raw_data
    from config.settings import settings

    target = args.target_bugs or settings.MAX_ISSUES
    print(f"Fetching {target} raw Bugzilla Core bugs...")

    raw_data = build_bugzilla_core_raw_data(target_count=target)
    if not raw_data:
        print("No Bugzilla records were produced. Check logs.")
        sys.exit(1)

    output_path = save_bugzilla_raw_data(raw_data)
    print(f"Done! {len(raw_data)} raw Bugzilla records saved to {output_path}")


def cmd_build_index(args) -> None:
    """Build clean dataset, split test set, run regex+spaCy, and index in ChromaDB."""
    from pipeline.index_builder import build_index

    summary = build_index(rebuild_clean=True, reset=True)

    stats = summary.get("collection_stats", {})
    print(
        f"\nDone! "
        f"total_clean={summary.get('total_clean', 0)}, "
        f"train={summary.get('train_count', 0)}, "
        f"test={summary.get('test_count', 0)}, "
        f"chroma_count={stats.get('count', 0)}"
    )
    if summary.get("test_output"):
        print(f"Test set saved to: {summary['test_output']}")


def cmd_query(_args) -> None:
    """Start interactive CLI query loop."""
    from modules.vector_store import get_collection_stats
    from pipeline.triage_pipeline import run_triage_pipeline

    try:
        stats = get_collection_stats()
    except Exception as exc:
        print(f"Failed to open ChromaDB: {exc}")
        sys.exit(1)

    count = int(stats.get("count", 0))
    if count == 0:
        print("Vector store is empty. Run these first:")
        print("  1) python main.py ingest")
        print("  2) python main.py build-index")
        sys.exit(1)

    print("=" * 72)
    print("Mozilla Core RAG Pipeline (Interactive)")
    print("Type your query and press Enter. Type 'exit' to quit.")
    print(f"Indexed documents: {count}")
    print("=" * 72)

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        result = run_triage_pipeline(
            title=query[:140],
            description=query,
            labels=["mozilla-core", "bug"],
            comments="",
        )

        classification = result.get("classification", {}) or {}
        generated = result.get("generated_response", {}) or {}
        references = result.get("retrieved_references", []) or []
        metadata = result.get("metadata", {}) or {}

        print("\n" + "=" * 72)
        print("RAG TRIAGE RESULT")
        print("=" * 72)
        print(
            f"Classification: type={classification.get('type', 'unknown')}, "
            f"severity={classification.get('severity', 'unknown')}, "
            f"team={classification.get('team', 'unknown')}"
        )

        print("\nRouting Explanation:")
        print(generated.get("routing_explanation", "N/A"))

        print("\nDebugging Steps:")
        for idx, step in enumerate(generated.get("debugging_steps", []), 1):
            print(f"  {idx}. {step}")

        print("\nPossible Causes:")
        for idx, cause in enumerate(generated.get("possible_causes", []), 1):
            print(f"  {idx}. {cause}")

        print("\nTop Retrieved References:")
        for ref in references[:5]:
            print(f"  - #{ref.get('issue_number', 'N/A')} | {ref.get('similarity_score', 0):.4f} | {ref.get('title', 'Unknown')}")

        if metadata.get("hyde_activated"):
            print("\n  [HyDE was activated for this query]")

        print(f"\nProcessing time: {metadata.get('processing_time_ms', 0)}ms")
        print("=" * 72)


def cmd_serve(args) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import settings

    print(f"Starting Triage API (Ollama/{settings.OLLAMA_MODEL})...")
    print(f"Swagger UI: http://localhost:{settings.API_PORT}/docs")

    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=args.reload,
        log_level="info",
    )


def cmd_ui(args) -> None:
    """Launch the Streamlit frontend."""
    port = args.port if hasattr(args, "port") else 8501
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(port)]
    subprocess.run(cmd, check=False)


def cmd_eval_dense(_args) -> None:
    """Run dense-only retrieval evaluation."""
    print("=" * 60)
    print("Running Dense-Only evaluation...")
    print("=" * 60)
    subprocess.run([sys.executable, "tests/dense_only_retrival/run_eval.py"], check=False)


def cmd_eval_sparse(_args) -> None:
    """Run sparse-only retrieval evaluation."""
    print("=" * 60)
    print("Running Sparse-Only evaluation...")
    print("=" * 60)
    subprocess.run([sys.executable, "tests/sparse_only_retrival/run_eval.py"], check=False)


def cmd_eval_hybrid(_args) -> None:
    """Run hybrid retrieval evaluation (with HyDE)."""
    print("=" * 60)
    print("Running Hybrid evaluation (HyDE enabled)...")
    print("=" * 60)
    subprocess.run([sys.executable, "tests/hybrid_retrival/run_eval.py"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mozilla Core Autonomous Ticket Triage (Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start (all with zero arguments):\n"
            "  python main.py ingest\n"
            "  python main.py build-index\n"
            "  python main.py query\n"
            "  python main.py ui\n"
            "\n"
            "Evaluation (run individually):\n"
            "  python main.py eval-dense\n"
            "  python main.py eval-sparse\n"
            "  python main.py eval-hybrid\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Ingest ----
    ingest_parser = subparsers.add_parser("ingest", help="Fetch raw Bugzilla Core bugs (default: 3000)")
    ingest_parser.add_argument("--target-bugs", type=int, default=None, help="Target number of bugs")

    # ---- Build Index ----
    subparsers.add_parser("build-index", help="Preprocess, split test set, build ChromaDB index")

    # ---- Interactive Query ----
    subparsers.add_parser("query", help="Open interactive terminal query loop")

    # ---- Serve API ----
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # ---- Streamlit UI ----
    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit frontend")
    ui_parser.add_argument("--port", type=int, default=8501, help="Streamlit port")

    # ---- Evaluations (individual) ----
    subparsers.add_parser("eval-dense", help="Evaluate with dense-only retrieval (no HyDE)")
    subparsers.add_parser("eval-sparse", help="Evaluate with sparse-only retrieval (no HyDE)")
    subparsers.add_parser("eval-hybrid", help="Evaluate with hybrid retrieval (HyDE enabled)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ingest": cmd_ingest,
        "build-index": cmd_build_index,
        "query": cmd_query,
        "serve": cmd_serve,
        "ui": cmd_ui,
        "eval-dense": cmd_eval_dense,
        "eval-sparse": cmd_eval_sparse,
        "eval-hybrid": cmd_eval_hybrid,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
