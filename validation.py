"""Shared validation for grocery-to-TickTick import requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImportRequest:
    image_path: Path
    project: str
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    dry_run: bool = False
    ticktick_access_token: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""


def validate_import_request(req: ImportRequest) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []
    provider = req.provider.strip().casefold()
    if not req.image_path.exists():
        errors.append(f"Image not found: {req.image_path}")
    if not req.project.strip():
        errors.append("Project name is required")
    if provider not in {"openai", "anthropic"}:
        errors.append('Provider must be one of: "openai", "anthropic"')
    elif provider == "openai" and not req.openai_api_key:
        errors.append("Missing OPENAI_API_KEY")
    elif provider == "anthropic" and not req.anthropic_api_key:
        errors.append("Missing ANTHROPIC_API_KEY")
    if not req.dry_run and not req.ticktick_access_token:
        errors.append("Missing TickTick access token")
    return errors
