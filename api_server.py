import os
import tempfile
import mimetypes
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from openai import OpenAIError

from grocery_to_ticktick import (
    TickTickError,
    extract_ingredients,
    sync_items_to_project,
)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="TickTick Grocery Import API")


def _get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: missing required env var {name}",
        )
    return value


def _get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


@app.post("/api/ticktick/import")
async def import_ticktick_tasks(
    image: UploadFile | None = File(default=None),
    project: str | None = Form(default=None),
    dry_run: bool = Form(default=False),
    model: str = Form(default=DEFAULT_MODEL),
    authorization: str | None = Header(default=None),
) -> dict:
    expected_api_key = _get_required_env("API_KEY")
    provided_token = _get_bearer_token(authorization)
    if provided_token != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if image is None:
        raise HTTPException(
            status_code=400, detail='Missing required file field "image"'
        )

    content_type = (image.content_type or "").strip().lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    default_project = os.environ.get("DEFAULT_TICKTICK_PROJECT", "").strip()
    project_name = (project or default_project).strip()

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    ticktick_access_token = os.environ.get("TICKTICK_ACCESS_TOKEN", "")
    early_errors: list[str] = []
    if not project_name:
        early_errors.append("Project name is required")
    if not openai_api_key:
        early_errors.append("Missing OPENAI_API_KEY")
    if not dry_run and not ticktick_access_token:
        early_errors.append("Missing TickTick access token")
    if early_errors:
        raise HTTPException(status_code=400, detail=early_errors)

    max_upload_bytes_str = os.environ.get(
        "MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)
    )
    try:
        max_upload_bytes = int(max_upload_bytes_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail="Server misconfigured: MAX_UPLOAD_BYTES"
        ) from exc
    if max_upload_bytes <= 0:
        raise HTTPException(
            status_code=500, detail="Server misconfigured: MAX_UPLOAD_BYTES"
        )

    suffix = Path(image.filename or "upload").suffix
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".img"
    temp_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            while True:
                chunk = await image.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(
                        status_code=413, detail="Uploaded file is too large"
                    )
                tmp.write(chunk)

        ingredients = extract_ingredients(temp_path, model)
        created: list[str] = []
        skipped: list[str] = []
        if not dry_run and ingredients:
            created, skipped = sync_items_to_project(
                ticktick_access_token, project_name, ingredients
            )

        return {
            "project": project_name,
            "ingredients": ingredients,
            "created": created,
            "skipped": skipped,
            "dry_run": dry_run,
        }
    except HTTPException:
        raise
    except (TickTickError, OpenAIError, RuntimeError, requests.RequestException) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if image is not None:
            await image.close()
        if temp_path and temp_path.exists():
            temp_path.unlink()
