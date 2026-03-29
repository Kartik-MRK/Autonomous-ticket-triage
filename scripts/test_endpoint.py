"""
Test Endpoint Script
======================
Sends a sample ticket to the FastAPI endpoint and prints the result.

Usage:
    python scripts/test_endpoint.py [--url http://localhost:8000]
"""

import sys
import json
import argparse

import requests


def main():
    parser = argparse.ArgumentParser(description="Test the triage API endpoint")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server",
    )

    args = parser.parse_args()

    # Sample ticket for testing
    sample_ticket = {
        "title": "VSCode crashes when opening large file",
        "description": (
            "The editor freezes and becomes completely unresponsive when trying to "
            "open files larger than 200MB. This happens consistently on Windows 11 "
            "with VSCode version 1.85. The application needs to be force-closed via "
            "Task Manager. Memory usage spikes to over 4GB before the freeze. "
            "This regression started after the latest update."
        ),
        "labels": ["bug", "editor", "performance"],
        "comments": "Multiple users have reported this issue after the 1.85 update.",
    }

    print("=" * 60)
    print("Testing Triage Endpoint")
    print("=" * 60)
    print(f"URL: {args.url}/triage")
    print(f"Ticket: {sample_ticket['title']}")
    print()

    try:
        response = requests.post(
            f"{args.url}/triage",
            json=sample_ticket,
            timeout=120,
        )

        print(f"Status Code: {response.status_code}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("=" * 60)
            print("TRIAGE RESULT")
            print("=" * 60)
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {args.url}")
        print("Make sure the server is running: python main.py serve")
    except Exception as e:
        print(f"ERROR: {e}")

    # Test health endpoint
    print()
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)

    try:
        health_response = requests.get(f"{args.url}/health", timeout=10)
        print(f"Status Code: {health_response.status_code}")
        print(json.dumps(health_response.json(), indent=2))
    except Exception as e:
        print(f"Health check failed: {e}")


if __name__ == "__main__":
    main()
