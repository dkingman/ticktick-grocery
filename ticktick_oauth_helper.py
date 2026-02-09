#!/usr/bin/env python3
"""Run TickTick OAuth locally and print export commands for tokens."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests

AUTHORIZE_URL = "https://ticktick.com/oauth/authorize"
TOKEN_URL = "https://ticktick.com/oauth/token"


@dataclass
class OAuthResult:
    code: str | None = None
    state: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TickTick OAuth token helper.")
    parser.add_argument("--client-id", required=True, help="TickTick OAuth client ID")
    parser.add_argument(
        "--client-secret", required=True, help="TickTick OAuth client secret"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local callback port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local callback host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--scope",
        default="tasks:read tasks:write",
        help='OAuth scope (default: "tasks:read tasks:write")',
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open auth URL in default browser automatically",
    )
    return parser.parse_args()


def make_handler(result: OAuthResult):
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
                    b"TickTick authorization received. You can close this tab."
                )
            else:
                self.wfile.write(
                    b"Authorization failed or missing code. Return to terminal."
                )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return CallbackHandler


def exchange_code_for_token(
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
        TOKEN_URL,
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
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()
    redirect_uri = f"http://{args.host}:{args.port}/callback"
    state = secrets.token_urlsafe(16)
    result = OAuthResult()

    params = {
        "scope": args.scope,
        "client_id": args.client_id,
        "state": state,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }
    auth_url = f"{AUTHORIZE_URL}?{urlencode(params, quote_via=quote)}"

    server = HTTPServer((args.host, args.port), make_handler(result))
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("1) Open this URL in your browser and authorize:")
    print(auth_url)
    print("")
    if args.open_browser:
        webbrowser.open(auth_url)
        print("Opened browser automatically.")
        print("")

    print("2) Waiting for callback...")
    thread.join(timeout=300)
    server.server_close()

    if result.error:
        print(f"OAuth error returned by TickTick: {result.error}")
        return 1
    if not result.code:
        print("No authorization code received. Confirm redirect URI in TickTick app settings.")
        return 1
    if result.state != state:
        print("State mismatch; aborting for safety.")
        return 1

    try:
        token_data = exchange_code_for_token(
            client_id=args.client_id,
            client_secret=args.client_secret,
            code=result.code,
            redirect_uri=redirect_uri,
            scope=args.scope,
        )
    except requests.RequestException as exc:
        print(f"Token exchange failed: {exc}")
        return 1

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in")

    if not access_token:
        print("Token response did not include access_token:")
        print(json.dumps(token_data, indent=2))
        return 1

    print("")
    print("Set these in your shell:")
    print(f'export TICKTICK_ACCESS_TOKEN="{access_token}"')
    if refresh_token:
        print(f'export TICKTICK_REFRESH_TOKEN="{refresh_token}"')
    if expires_in is not None:
        print(f"# access token expires in {expires_in} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

