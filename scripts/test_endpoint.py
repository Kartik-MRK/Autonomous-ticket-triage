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

    # Sample Mozilla Core style ticket for testing
    sample_ticket = {
        "title": "Firefox crashes during WebRTC media negotiation",
        "description": (
            "Firefox becomes unresponsive during WebRTC call setup on Linux after "
            "the media pipeline negotiates codecs. The browser UI freezes for 20-30 "
            "seconds and then crashes. This started in a recent nightly build."
        ),
        "labels": ["mozilla-core", "bug", "webrtc"],
        "comments": "Crash signatures appear around RTP transport negotiation.",
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
