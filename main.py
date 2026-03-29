"""
Autonomous Ticket Triage and Routing
========================================
Main entry point for the application.

Commands:
    python main.py serve          - Start the FastAPI server
    python main.py ingest         - Fetch issues from GitHub
    python main.py preprocess     - Preprocess raw issues
    python main.py build-index    - Build the vector store index
    python main.py evaluate       - Run evaluation metrics
    python main.py pipeline       - Run pipeline on a sample ticket
"""

import sys
import argparse


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from utils.config import settings

    print("Starting Autonomous Ticket Triage API Server...")
    print(f"Swagger UI: http://localhost:{settings.API_PORT}/docs")
    print(f"ReDoc: http://localhost:{settings.API_PORT}/redoc")

    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=args.reload,
        log_level="info",
    )


def cmd_ingest(args):
    """Run data ingestion from GitHub."""
    from modules.ingestion import fetch_issues, save_issues
    from utils.config import settings
    from utils.logger import get_logger

    logger = get_logger("main.ingest")

    if not settings.GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN is not set in .env file!")
        sys.exit(1)

    logger.info(f"Fetching up to {args.max_issues} issues from GitHub...")
    issues = fetch_issues(max_issues=args.max_issues)
    save_issues(issues)
    print(f"Done! {len(issues)} issues saved to {settings.RAW_ISSUES_FILE}")


def cmd_preprocess(args):
    """Preprocess raw issues."""
    from modules.ingestion import load_raw_issues
    from modules.preprocessing import preprocess_batch
    from utils.logger import get_logger

    logger = get_logger("main.preprocess")

    logger.info("Loading raw issues...")
    raw_issues = load_raw_issues()

    logger.info(f"Preprocessing {len(raw_issues)} issues...")
    processed = preprocess_batch(raw_issues)
    print(f"Done! {len(processed)} issues preprocessed and saved.")


def cmd_build_index(args):
    """Build the vector store index."""
    from modules.ingestion import load_raw_issues
    from modules.preprocessing import preprocess_batch, load_processed_issues
    from modules.embedding import generate_embeddings_batch
    from modules.vector_store import add_documents, get_collection, get_collection_stats
    from utils.config import settings
    from utils.logger import get_logger

    logger = get_logger("main.build_index")

    # Check if processed data exists
    if settings.PROCESSED_ISSUES_FILE.exists() and not args.reprocess:
        logger.info("Loading existing processed data...")
        processed_issues = load_processed_issues()
    else:
        logger.info("Loading and preprocessing raw issues...")
        raw_issues = load_raw_issues()
        processed_issues = preprocess_batch(raw_issues)

    if args.reset:
        logger.info("Resetting vector store...")
        get_collection(reset=True)

    # Generate embeddings
    logger.info("Generating embeddings (this may take a while)...")
    texts = [issue["unified_text"] for issue in processed_issues]
    embeddings = generate_embeddings_batch(texts, is_query=False, batch_size=32)

    # Store in ChromaDB
    ids = [str(issue["number"]) for issue in processed_issues]
    metadatas = [
        {
            "issue_number": str(issue["number"]),
            "title": issue.get("original_title", ""),
            "labels": ", ".join(issue.get("labels", [])),
            "state": issue.get("state", ""),
            "assignees": ", ".join(issue.get("assignees", [])),
        }
        for issue in processed_issues
    ]

    add_documents(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    stats = get_collection_stats()
    print(f"Done! Vector store contains {stats['count']} documents.")


def cmd_evaluate(args):
    """Run evaluation."""
    from evaluation.evaluate import run_evaluation

    results = run_evaluation(
        max_test_samples=args.max_samples,
    )

    print("\n=== Evaluation Summary ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, (int, str)):
            print(f"  {key}: {value}")


def cmd_pipeline(args):
    """Run the pipeline on a sample ticket."""
    import json
    from pipeline.triage_pipeline import run_triage_pipeline

    # Sample ticket
    result = run_triage_pipeline(
        title=args.title or "VSCode crashes when opening large file",
        description=args.description or (
            "The editor freezes and becomes completely unresponsive when trying to "
            "open files larger than 200MB. This happens consistently on Windows 11. "
            "Memory usage spikes to over 4GB before the freeze."
        ),
        labels=args.labels.split(",") if args.labels else ["bug", "editor"],
    )

    print("\n" + "=" * 60)
    print("TRIAGE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Ticket Triage and Routing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py serve                    # Start API server
  python main.py ingest --max-issues 500  # Fetch 500 GitHub issues
  python main.py preprocess               # Clean and tokenize issues
  python main.py build-index --reset      # Build vector store
  python main.py evaluate                 # Run evaluation metrics
  python main.py pipeline                 # Test with sample ticket
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Serve ----
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # ---- Ingest ----
    ingest_parser = subparsers.add_parser("ingest", help="Fetch issues from GitHub")
    ingest_parser.add_argument("--max-issues", type=int, default=500, help="Max issues to fetch")

    # ---- Preprocess ----
    subparsers.add_parser("preprocess", help="Preprocess raw issues")

    # ---- Build Index ----
    index_parser = subparsers.add_parser("build-index", help="Build the vector store index")
    index_parser.add_argument("--reset", action="store_true", help="Reset vector store first")
    index_parser.add_argument("--reprocess", action="store_true", help="Force reprocessing")

    # ---- Evaluate ----
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--max-samples", type=int, default=50, help="Max test samples")

    # ---- Pipeline ----
    pipeline_parser = subparsers.add_parser("pipeline", help="Run pipeline on a ticket")
    pipeline_parser.add_argument("--title", type=str, help="Ticket title")
    pipeline_parser.add_argument("--description", type=str, help="Ticket description")
    pipeline_parser.add_argument("--labels", type=str, help="Comma-separated labels")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to the appropriate command
    commands = {
        "serve": cmd_serve,
        "ingest": cmd_ingest,
        "preprocess": cmd_preprocess,
        "build-index": cmd_build_index,
        "evaluate": cmd_evaluate,
        "pipeline": cmd_pipeline,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
