#!/usr/bin/env python3
"""Extract grocery items from an image and add them to a TickTick list."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests
from openai import OpenAI, OpenAIError

TICKTICK_API_BASE = "https://api.ticktick.com/open/v1"
TICKTICK_AUTHORIZE_URL = "https://ticktick.com/oauth/authorize"
TICKTICK_TOKEN_URL = "https://ticktick.com/oauth/token"


class TickTickError(RuntimeError):
    """Raised when TickTick API calls fail."""


class OAuthFlowError(RuntimeError):
    """Raised when OAuth authentication fails."""


class _OAuthResult:
    def __init__(self) -> None:
        self.code: str | None = None
        self.state: str | None = None
        self.error: str | None = None


def _parse_json_from_model_text(text: str) -> dict:
    candidate = text.strip()

    # Handle fenced blocks like ```json ... ```
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first decodable JSON object in mixed text.
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"Model did not return valid JSON: {text}")


def _clean_items(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", value.strip())
        # Remove list bullets while preserving quantity prefixes like "2 cups ..."
        item = re.sub(r"^[\-\*\u2022]\s*", "", item)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def extract_ingredients(image_path: Path, model: str) -> list[str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise RuntimeError(f"Unsupported image type for file: {image_path}")

    prompt = (
        "Extract grocery ingredients from this image. "
        "Return strict JSON in this exact shape: "
        '{"ingredients": ["item 1", "item 2"]}. '
        "Only include buyable grocery items, deduplicated. "
        "Keep quantities and units exactly when present (example: '2 cups beef broth'). "
        "Remove recipe instructions only."
        "Skip salt, pepper, and water if they are in the list of ingredients."
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            }
        ],
    )

    text = response.output_text.strip()
    payload = _parse_json_from_model_text(text)

    raw_items = payload.get("ingredients", [])
    if not isinstance(raw_items, list):
        raise RuntimeError("Unexpected JSON shape from model output")

    return _clean_items(str(item) for item in raw_items)


def _ticktick_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _ticktick_request(
    method: str,
    path: str,
    access_token: str,
    json_payload: dict | None = None,
) -> dict | list:
    url = f"{TICKTICK_API_BASE}{path}"
    response = requests.request(
        method,
        url,
        headers=_ticktick_headers(access_token),
        json=json_payload,
        timeout=30,
    )
    if response.status_code >= 400:
        raise TickTickError(
            f"TickTick API {method} {path} failed ({response.status_code}): {response.text}"
        )
    if not response.text:
        return {}
    return response.json()


def _make_callback_handler(result: _OAuthResult):
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            query = parse_qs(urlparse(self.path).query)
            result.code = query.get("code", [None])[0]
            result.state = query.get("state", [None])[0]
            result.error = query.get("error", [None])[0]

            ok = result.code is not None and result.error is None
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            if ok:
                self.wfile.write(
                    b"TickTick authorization received. Return to terminal."
                )
            else:
                self.wfile.write(
                    b"TickTick authorization failed. Return to terminal."
                )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return CallbackHandler


def _exchange_code_for_token(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    scope: str,
) -> dict:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode(
        "utf-8"
    )
    response = requests.post(
        TICKTICK_TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code,
            "grant_type": "authorization_code",
            "scope": scope,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise OAuthFlowError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )
    return response.json()


def get_or_create_ticktick_token(args: argparse.Namespace) -> str:
    if args.ticktick_access_token:
        return args.ticktick_access_token

    client_id = args.ticktick_client_id or os.environ.get("TICKTICK_CLIENT_ID")
    client_secret = args.ticktick_client_secret or os.environ.get(
        "TICKTICK_CLIENT_SECRET"
    )
    if not client_id or not client_secret:
        raise OAuthFlowError(
            "Missing TickTick auth. Set TICKTICK_ACCESS_TOKEN or provide "
            "--ticktick-client-id and --ticktick-client-secret."
        )

    redirect_uri = f"http://{args.oauth_host}:{args.oauth_port}/callback"
    state = secrets.token_urlsafe(16)
    result = _OAuthResult()
    auth_url = (
        f"{TICKTICK_AUTHORIZE_URL}?"
        f"{urlencode({'scope': args.oauth_scope, 'client_id': client_id, 'state': state, 'redirect_uri': redirect_uri, 'response_type': 'code'}, quote_via=quote)}"
    )

    server = HTTPServer((args.oauth_host, args.oauth_port), _make_callback_handler(result))
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("Open this URL and authorize TickTick:")
    print(auth_url)
    if args.oauth_open_browser:
        webbrowser.open(auth_url)
        print("Opened browser automatically.")
    print("Waiting for OAuth callback...")

    thread.join(timeout=args.oauth_timeout_seconds)
    server.server_close()

    if result.error:
        raise OAuthFlowError(f"TickTick OAuth error: {result.error}")
    if not result.code:
        raise OAuthFlowError(
            "No OAuth code received. Check your TickTick redirect URI setting."
        )
    if result.state != state:
        raise OAuthFlowError("OAuth state mismatch.")

    token_data = _exchange_code_for_token(
        client_id=client_id,
        client_secret=client_secret,
        code=result.code,
        redirect_uri=redirect_uri,
        scope=args.oauth_scope,
    )
    access_token = str(token_data.get("access_token", "")).strip()
    if not access_token:
        raise OAuthFlowError(f"Missing access_token in response: {token_data}")

    refresh_token = str(token_data.get("refresh_token", "")).strip()
    expires_in = token_data.get("expires_in")
    print("Auth complete. Save these in your shell to skip OAuth next time:")
    print(f'export TICKTICK_ACCESS_TOKEN="{access_token}"')
    if refresh_token:
        print(f'export TICKTICK_REFRESH_TOKEN="{refresh_token}"')
    else:
        print("TickTick did not return a refresh token in this response.")
    if expires_in is not None:
        print(f"# expires_in: {expires_in} seconds")
    return access_token


def get_project_id(access_token: str, project_name: str) -> str:
    projects = _ticktick_request("GET", "/project", access_token)
    if not isinstance(projects, list):
        raise TickTickError("Unexpected /project response")

    for project in projects:
        if str(project.get("name", "")).casefold() == project_name.casefold():
            return str(project["id"])

    raise TickTickError(
        f'Project "{project_name}" not found. Create it in TickTick first, or pass --project.'
    )


def get_existing_task_titles(access_token: str, project_id: str) -> set[str]:
    project_data = _ticktick_request("GET", f"/project/{project_id}/data", access_token)
    tasks = project_data.get("tasks", []) if isinstance(project_data, dict) else []
    return {str(task.get("title", "")).casefold() for task in tasks}


def create_task(access_token: str, title: str, project_id: str) -> dict:
    payload = {
        "title": title,
        "projectId": project_id,
    }
    result = _ticktick_request("POST", "/task", access_token, payload)
    if not isinstance(result, dict):
        raise TickTickError("Unexpected /task response")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ingredients from an image and add them to TickTick."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--project",
        required=True,
        help="TickTick project/list name (required)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Vision model to use for extraction (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted ingredients only; skip TickTick auth/API calls",
    )
    parser.add_argument(
        "--ticktick-access-token",
        default=os.environ.get("TICKTICK_ACCESS_TOKEN", ""),
        help="TickTick bearer token (or env TICKTICK_ACCESS_TOKEN)",
    )
    parser.add_argument(
        "--ticktick-client-id",
        default=os.environ.get("TICKTICK_CLIENT_ID", ""),
        help="TickTick OAuth client ID (or env TICKTICK_CLIENT_ID)",
    )
    parser.add_argument(
        "--ticktick-client-secret",
        default=os.environ.get("TICKTICK_CLIENT_SECRET", ""),
        help="TickTick OAuth client secret (or env TICKTICK_CLIENT_SECRET)",
    )
    parser.add_argument(
        "--oauth-host",
        default="127.0.0.1",
        help="OAuth callback host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--oauth-port",
        type=int,
        default=8765,
        help="OAuth callback port (default: 8765)",
    )
    parser.add_argument(
        "--oauth-scope",
        default="tasks:read tasks:write",
        help='OAuth scope (default: "tasks:read tasks:write")',
    )
    parser.add_argument(
        "--oauth-timeout-seconds",
        type=int,
        default=300,
        help="OAuth callback timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--oauth-open-browser",
        action="store_true",
        help="Open TickTick auth URL automatically",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing env var: OPENAI_API_KEY", file=sys.stderr)
        return 1

    try:
        items = extract_ingredients(args.image, args.model)
        if not items:
            print("No grocery items found in the image.")
            return 0

        if args.dry_run:
            for item in items:
                print(item)
            return 0

        ticktick_access_token = get_or_create_ticktick_token(args)
        print(f"Extracted {len(items)} items from image:")
        for item in items:
            print(f"- {item}")

        project_id = get_project_id(ticktick_access_token, args.project)
        existing = get_existing_task_titles(ticktick_access_token, project_id)

        created = 0
        skipped = 0
        for item in items:
            if item.casefold() in existing:
                skipped += 1
                continue
            create_task(ticktick_access_token, item, project_id)
            existing.add(item.casefold())
            created += 1

        print(f"Done. Created {created} tasks, skipped {skipped} duplicates.")
        return 0

    except (
        TickTickError,
        RuntimeError,
        OpenAIError,
        requests.RequestException,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
