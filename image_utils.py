"""Helpers for normalizing images before sending to OpenAI."""

from __future__ import annotations

import mimetypes
import tempfile
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

HEIC_MIME_TYPES = {"image/heic", "image/heif"}
HEIC_EXTENSIONS = {".heic", ".heif"}


def _register_heic_mimetypes() -> None:
    mimetypes.add_type("image/heic", ".heic")
    mimetypes.add_type("image/heif", ".heif")


def _is_heic(path: Path, content_type: str | None = None) -> bool:
    if content_type and content_type.strip().lower() in HEIC_MIME_TYPES:
        return True
    return path.suffix.lower() in HEIC_EXTENSIONS


def normalize_image_for_openai(
    image_path: Path, content_type: str | None = None
) -> tuple[Path, list[Path]]:
    """Return a path OpenAI accepts plus any temp files to clean up."""
    _register_heic_mimetypes()
    if not _is_heic(image_path, content_type):
        return image_path, []

    pillow_heif.register_heif_opener()
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = Path(tmp.name)
        image.save(temp_path, format="JPEG", quality=90, optimize=True)

    return temp_path, [temp_path]
