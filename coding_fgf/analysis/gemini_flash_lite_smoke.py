from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import google.auth.transport.requests
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
import requests


DEFAULT_CREDENTIALS = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/gcloud/application_default_credentials.json"))
DEFAULT_PROJECT = "project_o"
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def load_credentials(path: Path) -> Credentials | service_account.Credentials:
    credentials, _ = google.auth.load_credentials_from_file(
        str(path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials


def generate_content(
    *,
    credentials_path: Path = DEFAULT_CREDENTIALS,
    project_id: str = DEFAULT_PROJECT,
    location: str = DEFAULT_LOCATION,
    model: str = DEFAULT_MODEL,
    prompt: str = "Reply with exactly: gemini smoke test ok",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    credentials = load_credentials(credentials_path)
    model_resource = f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"
    url = f"https://aiplatform.googleapis.com/v1/{model_resource}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 32,
        },
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def response_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if "text" in part:
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Vertex AI Gemini Flash-Lite smoke call.")
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Reply with exactly: gemini smoke test ok")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Print the full JSON response.")
    args = parser.parse_args()

    response = generate_content(
        credentials_path=args.credentials,
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(response_text(response))


if __name__ == "__main__":
    main()
