"""
Run Bugzilla RAG Pipeline (Interactive CLI)
==========================================
Starts an interactive terminal loop for querying the Mozilla Core
RAG pipeline against the prepared ChromaDB index.

Usage:
    python scripts/run_bugzilla_pipeline.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.vector_store import get_collection_stats
from pipeline.triage_pipeline import run_triage_pipeline


def _print_result(result: dict) -> None:
    """Print a concise, readable summary of the pipeline result."""
    classification = result.get("classification", {}) or {}
    generated = result.get("generated_response", {}) or {}
    references = result.get("retrieved_references", []) or []
    metadata = result.get("metadata", {}) or {}

    print("\n" + "=" * 72)
    print("RAG TRIAGE RESULT")
    print("=" * 72)
    print(
        "Classification: "
        f"type={classification.get('type', 'unknown')}, "
        f"severity={classification.get('severity', 'unknown')}, "
        f"team={classification.get('team', 'unknown')}"
    )

    print("\nRouting Explanation:")
    print(generated.get("routing_explanation", "N/A"))

    print("\nDebugging Steps:")
    for idx, step in enumerate(generated.get("debugging_steps", []), start=1):
        print(f"{idx}. {step}")

    print("\nPossible Causes:")
    for idx, cause in enumerate(generated.get("possible_causes", []), start=1):
        print(f"{idx}. {cause}")

    print("\nTop Retrieved References:")
    if not references:
        print("No references found")
    else:
        for ref in references[:5]:
            issue_number = ref.get("issue_number", "N/A")
            title = ref.get("title", "Unknown")
            score = ref.get("similarity_score", 0.0)
            print(f"- #{issue_number} | {score:.4f} | {title}")

    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))
    print("=" * 72 + "\n")


def main() -> None:
    try:
        stats = get_collection_stats()
    except Exception as exc:
        print(f"Failed to open ChromaDB: {exc}")
        raise SystemExit(1)

    count = int(stats.get("count", 0))
    if count == 0:
        print(
            "Vector store is empty. Run these first:\n"
            "1) python scripts/ingest_bugzilla_core.py --target-bugs 800\n"
            "2) python scripts/build_bugzilla_index.py --rebuild-clean --reset"
        )
        raise SystemExit(1)

    print("=" * 72)
    print("Mozilla Core RAG Pipeline (Interactive)")
    print("Type your query and press Enter. Type 'exit' to quit.")
    print(f"Indexed documents available: {count}")
    print("=" * 72)

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive pipeline.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive pipeline.")
            break

        result = run_triage_pipeline(
            title=query[:140],
            description=query,
            labels=["mozilla-core", "bug"],
            comments="",
        )
        _print_result(result)


if __name__ == "__main__":
    main()
