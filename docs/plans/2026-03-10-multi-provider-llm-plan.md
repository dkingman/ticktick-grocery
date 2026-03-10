# Multi-Provider LLM Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Anthropic Claude as an alternative vision provider alongside OpenAI, selectable via `LLM_PROVIDER` env var.

**Architecture:** New `providers.py` module with a factory pattern. Each provider is a function that takes `(image_b64, mime_type, prompt, model)` and returns raw text. `extract_ingredients()` delegates to the selected provider. Unified `ProviderError` replaces provider-specific error handling everywhere.

**Tech Stack:** Python, openai SDK, anthropic SDK, pytest, FastAPI

---

### Task 1: Add anthropic dependency

**Files:**
- Modify: `pyproject.toml:6-14`

**Step 1: Add the dependency**

In `pyproject.toml`, add `"anthropic>=0.40.0"` to the `dependencies` list:

```python
dependencies = [
  "anthropic>=0.40.0",
  "fastapi>=0.116.1",
  "openai>=1.60.0",
  "pillow>=10.0.0",
  "pillow-heif>=0.13.0",
  "python-multipart>=0.0.20",
  "requests>=2.31.0",
  "uvicorn[standard]>=0.35.0",
]
```

**Step 2: Install**

Run: `uv sync`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add anthropic SDK dependency"
```

---

### Task 2: Create `providers.py` with tests (TDD)

**Files:**
- Create: `providers.py`
- Create: `tests/test_providers.py`

**Step 1: Write failing tests for helper functions**

Create `tests/test_providers.py`:

```python
import os
import unittest
from unittest.mock import MagicMock, patch

import providers


class GetProviderNameTests(unittest.TestCase):
    def test_defaults_to_openai(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(providers.get_provider_name(), "openai")

    def test_reads_env_var(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            self.assertEqual(providers.get_provider_name(), "anthropic")

    def test_normalizes_to_lowercase(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "Anthropic"}, clear=True):
            self.assertEqual(providers.get_provider_name(), "anthropic")


class GetDefaultModelTests(unittest.TestCase):
    def test_openai_default(self) -> None:
        self.assertEqual(providers.get_default_model("openai"), "gpt-4.1-mini")

    def test_anthropic_default(self) -> None:
        self.assertEqual(
            providers.get_default_model("anthropic"), "claude-sonnet-4-20250514"
        )

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(providers.ProviderError):
            providers.get_default_model("gemini")


class GetApiKeyEnvVarTests(unittest.TestCase):
    def test_openai(self) -> None:
        self.assertEqual(providers.get_api_key_env_var("openai"), "OPENAI_API_KEY")

    def test_anthropic(self) -> None:
        self.assertEqual(
            providers.get_api_key_env_var("anthropic"), "ANTHROPIC_API_KEY"
        )

    def test_unknown_raises(self) -> None:
        with self.assertRaises(providers.ProviderError):
            providers.get_api_key_env_var("gemini")


class GetProviderTests(unittest.TestCase):
    def test_returns_openai_callable(self) -> None:
        fn = providers.get_provider("openai")
        self.assertEqual(fn, providers.extract_with_openai)

    def test_returns_anthropic_callable(self) -> None:
        fn = providers.get_provider("anthropic")
        self.assertEqual(fn, providers.extract_with_anthropic)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(providers.ProviderError):
            providers.get_provider("gemini")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers'`

**Step 3: Implement helper functions in `providers.py`**

Create `providers.py`:

```python
"""Pluggable LLM provider abstraction for vision extraction."""

import logging
import os

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""


DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-20250514",
}

API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", "openai").strip().lower()


def get_default_model(provider: str) -> str:
    if provider not in DEFAULT_MODELS:
        raise ProviderError(f"Unknown provider: {provider}")
    return DEFAULT_MODELS[provider]


def get_api_key_env_var(provider: str) -> str:
    if provider not in API_KEY_ENV_VARS:
        raise ProviderError(f"Unknown provider: {provider}")
    return API_KEY_ENV_VARS[provider]


def extract_with_openai(
    image_b64: str, mime_type: str, prompt: str, model: str
) -> str:
    raise NotImplementedError


def extract_with_anthropic(
    image_b64: str, mime_type: str, prompt: str, model: str
) -> str:
    raise NotImplementedError


_PROVIDERS = {
    "openai": extract_with_openai,
    "anthropic": extract_with_anthropic,
}


def get_provider(name: str):
    if name not in _PROVIDERS:
        raise ProviderError(f"Unknown provider: {name}")
    return _PROVIDERS[name]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: add providers module with helpers and tests"
```

---

### Task 3: Implement `extract_with_openai` with tests (TDD)

**Files:**
- Modify: `providers.py`
- Modify: `tests/test_providers.py`

**Step 1: Write failing tests for OpenAI extraction**

Append to `tests/test_providers.py`:

```python
class ExtractWithOpenaiTests(unittest.TestCase):
    def test_calls_openai_client_correctly(self) -> None:
        mock_response = MagicMock()
        mock_response.output_text = "extracted text"
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch("providers.OpenAI", return_value=mock_client) as mock_cls:
                result = providers.extract_with_openai(
                    "aW1hZ2U=", "image/png", "Extract items", "gpt-4.1-mini"
                )

        mock_cls.assert_called_once_with(api_key="test-key")
        call_kwargs = mock_client.responses.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4.1-mini")
        content = call_kwargs["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[0]["text"], "Extract items")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertIn("data:image/png;base64,aW1hZ2U=", content[1]["image_url"])
        self.assertEqual(result, "extracted text")

    def test_wraps_openai_error_in_provider_error(self) -> None:
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("boom")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch("providers.OpenAI", return_value=mock_client):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.extract_with_openai(
                        "aW1hZ2U=", "image/png", "prompt", "gpt-4.1-mini"
                    )
        self.assertIn("boom", str(ctx.exception))

    def test_missing_api_key_raises_provider_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(providers.ProviderError):
                providers.extract_with_openai(
                    "aW1hZ2U=", "image/png", "prompt", "gpt-4.1-mini"
                )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers.py::ExtractWithOpenaiTests -v`
Expected: FAIL — `NotImplementedError`

**Step 3: Implement `extract_with_openai`**

Replace the stub in `providers.py`:

```python
def extract_with_openai(
    image_b64: str, mime_type: str, prompt: str, model: str
) -> str:
    from openai import OpenAI, OpenAIError

    key_var = API_KEY_ENV_VARS["openai"]
    api_key = os.environ.get(key_var, "").strip()
    if not api_key:
        raise ProviderError(f"Missing {key_var}")

    try:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI request start model=%s", model)
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
        return response.output_text
    except OpenAIError as exc:
        raise ProviderError(str(exc)) from exc
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: implement extract_with_openai in providers module"
```

---

### Task 4: Implement `extract_with_anthropic` with tests (TDD)

**Files:**
- Modify: `providers.py`
- Modify: `tests/test_providers.py`

**Step 1: Write failing tests for Anthropic extraction**

Append to `tests/test_providers.py`:

```python
class ExtractWithAnthropicTests(unittest.TestCase):
    def test_calls_anthropic_client_correctly(self) -> None:
        mock_block = MagicMock()
        mock_block.text = "extracted text"
        mock_message = MagicMock()
        mock_message.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("providers.Anthropic", return_value=mock_client) as mock_cls:
                result = providers.extract_with_anthropic(
                    "aW1hZ2U=", "image/png", "Extract items", "claude-sonnet-4-20250514"
                )

        mock_cls.assert_called_once_with(api_key="test-key")
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-sonnet-4-20250514")
        self.assertEqual(call_kwargs["max_tokens"], 1024)
        content = call_kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["source"]["type"], "base64")
        self.assertEqual(content[0]["source"]["media_type"], "image/png")
        self.assertEqual(content[0]["source"]["data"], "aW1hZ2U=")
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], "Extract items")
        self.assertEqual(result, "extracted text")

    def test_wraps_anthropic_error_in_provider_error(self) -> None:
        from anthropic import APIError

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = APIError(
            message="boom",
            request=MagicMock(),
            body=None,
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("providers.Anthropic", return_value=mock_client):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.extract_with_anthropic(
                        "aW1hZ2U=", "image/png", "prompt", "claude-sonnet-4-20250514"
                    )
        self.assertIn("boom", str(ctx.exception))

    def test_missing_api_key_raises_provider_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(providers.ProviderError):
                providers.extract_with_anthropic(
                    "aW1hZ2U=", "image/png", "prompt", "claude-sonnet-4-20250514"
                )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers.py::ExtractWithAnthropicTests -v`
Expected: FAIL — `NotImplementedError`

**Step 3: Implement `extract_with_anthropic`**

Replace the stub in `providers.py`:

```python
def extract_with_anthropic(
    image_b64: str, mime_type: str, prompt: str, model: str
) -> str:
    from anthropic import Anthropic, APIError

    key_var = API_KEY_ENV_VARS["anthropic"]
    api_key = os.environ.get(key_var, "").strip()
    if not api_key:
        raise ProviderError(f"Missing {key_var}")

    try:
        client = Anthropic(api_key=api_key)
        logger.info("Anthropic request start model=%s", model)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text
    except APIError as exc:
        raise ProviderError(str(exc)) from exc
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: implement extract_with_anthropic in providers module"
```

---

### Task 5: Rename `normalize_image_for_openai` to `normalize_image`

**Files:**
- Modify: `image_utils.py:27-30`
- Modify: `grocery_to_ticktick.py:21,97`
- Modify: `tests/test_grocery_to_ticktick.py:91`

**Step 1: Rename in `image_utils.py`**

Change line 27 docstring and function name:

```python
def normalize_image(
    image_path: Path, content_type: str | None = None
) -> tuple[Path, list[Path]]:
    """Return an image path suitable for LLM vision APIs plus any temp files to clean up."""
```

Also update the module docstring on line 1:

```python
"""Helpers for normalizing images before sending to vision LLM providers."""
```

**Step 2: Update import in `grocery_to_ticktick.py`**

Line 21: change `from image_utils import normalize_image_for_openai` to `from image_utils import normalize_image`

Line 97: change `normalize_image_for_openai(image_path)` to `normalize_image(image_path)`

**Step 3: Update test mock**

In `tests/test_grocery_to_ticktick.py` line 91: change `"grocery_to_ticktick.normalize_image_for_openai"` to `"grocery_to_ticktick.normalize_image"`

**Step 4: Run all tests**

Run: `pytest -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add image_utils.py grocery_to_ticktick.py tests/test_grocery_to_ticktick.py
git commit -m "refactor: rename normalize_image_for_openai to normalize_image"
```

---

### Task 6: Update `validation.py` to be provider-agnostic

**Files:**
- Modify: `validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Update test expectations first**

In `tests/test_validation.py`:

- Change `_valid_request` helper: rename `openai_api_key="sk-test"` to `llm_api_key="sk-test"`
- Change `test_missing_openai_api_key_returns_error` to `test_missing_llm_api_key_returns_error`, assert error message contains `"API key"` (not `"OPENAI_API_KEY"`)
- Change `test_multiple_errors_accumulate`: rename kwarg to `llm_api_key=""`

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'llm_api_key'`

**Step 3: Update `validation.py`**

```python
@dataclass
class ImportRequest:
    image_path: Path
    project: str
    model: str = "gpt-4.1-mini"
    dry_run: bool = False
    ticktick_access_token: str = ""
    llm_api_key: str = ""


def validate_import_request(req: ImportRequest) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []
    if not req.image_path.exists():
        errors.append(f"Image not found: {req.image_path}")
    if not req.project.strip():
        errors.append("Project name is required")
    if not req.llm_api_key:
        errors.append("Missing LLM API key")
    if not req.dry_run and not req.ticktick_access_token:
        errors.append("Missing TickTick access token")
    return errors
```

**Step 4: Run validation tests**

Run: `pytest tests/test_validation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add validation.py tests/test_validation.py
git commit -m "refactor: rename openai_api_key to llm_api_key in validation"
```

---

### Task 7: Wire providers into `grocery_to_ticktick.py` and update its tests

**Files:**
- Modify: `grocery_to_ticktick.py:1-18,94-151,382-396,447-498`
- Modify: `tests/test_grocery_to_ticktick.py`

**Step 1: Update tests first**

In `tests/test_grocery_to_ticktick.py`:

Replace `from openai import OpenAIError` with `from providers import ProviderError`.

Update `ExtractIngredientsTests.test_uses_detected_image_mime_type_in_data_url`:
- Remove `FakeClient` class and `patch("grocery_to_ticktick.OpenAI", ...)`.
- Instead, mock `grocery_to_ticktick.providers.get_provider_name` to return `"openai"` and mock `grocery_to_ticktick.providers.get_provider` to return a callable that captures args and returns the JSON text.
- Remove `patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})` — the provider module handles keys internally.

```python
class ExtractIngredientsTests(unittest.TestCase):
    def test_uses_detected_image_mime_type_in_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"fake-png-bytes")

            captured: dict = {}

            def fake_provider(image_b64, mime_type, prompt, model):
                captured["mime_type"] = mime_type
                captured["image_b64"] = image_b64
                return '{"ingredients":["milk"]}'

            with patch("grocery_to_ticktick.get_provider_name", return_value="openai"):
                with patch("grocery_to_ticktick.get_provider", return_value=fake_provider):
                    items = app.extract_ingredients(image_path, "gpt-4.1-mini")

            self.assertEqual(items, ["milk"])
            self.assertEqual(captured["mime_type"], "image/png")

    def test_heic_is_converted_to_jpeg_before_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            heic_path = Path(tmp) / "input.heic"
            heic_path.write_bytes(b"fake-heic-bytes")
            jpeg_path = Path(tmp) / "converted.jpg"
            jpeg_path.write_bytes(b"fake-jpg-bytes")

            captured: dict = {}

            def fake_provider(image_b64, mime_type, prompt, model):
                captured["mime_type"] = mime_type
                return '{"ingredients":["milk"]}'

            with patch("grocery_to_ticktick.get_provider_name", return_value="openai"):
                with patch("grocery_to_ticktick.get_provider", return_value=fake_provider):
                    with patch(
                        "grocery_to_ticktick.normalize_image",
                        return_value=(jpeg_path, []),
                    ) as mock_normalize:
                        items = app.extract_ingredients(heic_path, "gpt-4.1-mini")

            self.assertEqual(items, ["milk"])
            mock_normalize.assert_called_once_with(heic_path)
            self.assertEqual(captured["mime_type"], "image/jpeg")

    def test_rejects_unsupported_non_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.txt"
            image_path.write_text("not an image", encoding="utf-8")

            with patch("grocery_to_ticktick.get_provider_name", return_value="openai"):
                with patch("grocery_to_ticktick.get_provider"):
                    with self.assertRaises(RuntimeError):
                        app.extract_ingredients(image_path, "gpt-4.1-mini")
```

Update `MainErrorHandlingTests`:
- Replace `OpenAIError("boom")` with `ProviderError("boom")`.
- Remove `patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})`.

```python
class MainErrorHandlingTests(unittest.TestCase):
    def test_main_returns_one_on_provider_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.jpg"
            image_path.write_bytes(b"fake-jpg-bytes")
            args = Namespace(
                image=image_path,
                project="Grocery",
                model="gpt-4.1-mini",
                dry_run=False,
                ticktick_access_token="test-token",
                ticktick_client_id="",
                ticktick_client_secret="",
                oauth_host="127.0.0.1",
                oauth_port=8765,
                oauth_scope="tasks:read tasks:write",
                oauth_timeout_seconds=300,
                oauth_open_browser=False,
            )

            with patch("grocery_to_ticktick.parse_args", return_value=args):
                with patch(
                    "grocery_to_ticktick.extract_ingredients",
                    side_effect=ProviderError("boom"),
                ):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        exit_code = app.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("Error: boom", stderr.getvalue())
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grocery_to_ticktick.py -v`
Expected: FAIL — imports/mocks don't match current code

**Step 3: Update `grocery_to_ticktick.py`**

Replace lines 17-22 (imports):

```python
import requests

from logging_setup import configure_logging
from image_utils import normalize_image
from providers import ProviderError, get_api_key_env_var, get_default_model, get_provider, get_provider_name
from validation import ImportRequest, validate_import_request
```

(Remove `from openai import OpenAI, OpenAIError`.)

Replace `extract_ingredients` function (lines 94-151):

```python
def extract_ingredients(image_path: Path, model: str) -> list[str]:
    provider_name = get_provider_name()
    provider_fn = get_provider(provider_name)

    normalized_path, cleanup_paths = normalize_image(image_path)
    try:
        image_bytes = normalized_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(str(normalized_path))
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

        logger.info(
            "LLM request start provider=%s model=%s image=%s mime=%s bytes=%s",
            provider_name,
            model,
            normalized_path.name,
            mime_type,
            len(image_bytes),
        )
        text = provider_fn(image_b64, mime_type, prompt, model).strip()
        payload = _parse_json_from_model_text(text)

        raw_items = payload.get("ingredients", [])
        if not isinstance(raw_items, list):
            raise RuntimeError("Unexpected JSON shape from model output")

        return _clean_items(str(item) for item in raw_items)
    finally:
        for cleanup_path in cleanup_paths:
            try:
                cleanup_path.unlink()
            except FileNotFoundError:
                pass
```

Update `parse_args` (line 392-396) — make `--model` default provider-aware:

```python
    parser.add_argument(
        "--model",
        default=None,
        help="Vision model (default: provider-specific, e.g. gpt-4.1-mini for openai)",
    )
```

Update `main()` — replace `OpenAIError` with `ProviderError`, make model default provider-aware, update `ImportRequest` field name:

In `main()`, after `parse_args()` and before `ImportRequest(...)`:

```python
    provider_name = get_provider_name()
    if args.model is None:
        args.model = get_default_model(provider_name)
```

Change the `ImportRequest` construction to use `llm_api_key`:

```python
        req = ImportRequest(
            image_path=args.image,
            project=args.project,
            model=args.model,
            dry_run=args.dry_run,
            ticktick_access_token=ticktick_access_token,
            llm_api_key=os.environ.get(get_api_key_env_var(provider_name), ""),
        )
```

Replace exception tuple in `main()`:

```python
    except (
        TickTickError,
        RuntimeError,
        ProviderError,
        requests.RequestException,
    ) as exc:
```

**Step 4: Run tests**

Run: `pytest tests/test_grocery_to_ticktick.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add grocery_to_ticktick.py tests/test_grocery_to_ticktick.py
git commit -m "feat: wire providers into extract_ingredients and CLI"
```

---

### Task 8: Wire providers into `api_server.py` and update its tests

**Files:**
- Modify: `api_server.py`
- Modify: `tests/test_api_server.py`

**Step 1: Update tests first**

In `tests/test_api_server.py`:

Replace `from openai import OpenAIError` with `from providers import ProviderError`.

Update `base_env` in `setUp`: replace `"OPENAI_API_KEY": "openai-key"` with the provider-aware key. Since default provider is `openai`, keep `OPENAI_API_KEY` but remove the hardcoded name from the test structure:

```python
    def setUp(self) -> None:
        self.client = TestClient(api_server.app)
        self.base_env = {
            "API_KEY": "secret-key",
            "TICKTICK_ACCESS_TOKEN": "ticktick-token",
            "OPENAI_API_KEY": "openai-key",
            "DEFAULT_TICKTICK_PROJECT": "Default Errands",
            "MAX_UPLOAD_BYTES": "52428800",
        }
```

Update `test_upstream_failures_map_to_502` to use `ProviderError`:

```python
    def test_upstream_failures_map_to_502(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            with patch(
                "api_server.extract_ingredients", side_effect=ProviderError("boom")
            ):
                response = self.client.post(
                    "/api/ticktick/import",
                    headers=self._auth_headers(),
                    files={"image": ("list.png", b"fake", "image/png")},
                )
        self.assertEqual(response.status_code, 502)
        self.assertIn("boom", response.json()["detail"])
```

Add a test for Anthropic provider via env:

```python
    def test_anthropic_provider_validates_anthropic_key(self) -> None:
        env = dict(self.base_env)
        env["LLM_PROVIDER"] = "anthropic"
        env["ANTHROPIC_API_KEY"] = "ant-key"
        del env["OPENAI_API_KEY"]
        with patch.dict(os.environ, env, clear=True):
            with patch("api_server.extract_ingredients", return_value=["Milk"]):
                response = self.client.post(
                    "/api/ticktick/import",
                    headers=self._auth_headers(),
                    files={"image": ("list.png", b"fake", "image/png")},
                    data={"dry_run": "true"},
                )
        self.assertEqual(response.status_code, 200)

    def test_missing_provider_api_key_returns_400(self) -> None:
        env = dict(self.base_env)
        env["LLM_PROVIDER"] = "anthropic"
        del env["OPENAI_API_KEY"]
        # No ANTHROPIC_API_KEY set
        with patch.dict(os.environ, env, clear=True):
            response = self.client.post(
                "/api/ticktick/import",
                headers=self._auth_headers(),
                files={"image": ("list.png", b"fake", "image/png")},
            )
        self.assertEqual(response.status_code, 400)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_server.py -v`
Expected: FAIL

**Step 3: Update `api_server.py`**

Replace imports (lines 1-17):

```python
import os
import tempfile
import mimetypes
import logging
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from grocery_to_ticktick import (
    TickTickError,
    extract_ingredients,
    sync_items_to_project,
)
from providers import ProviderError, get_api_key_env_var, get_default_model, get_provider_name
```

Replace the `DEFAULT_MODEL` constant and use a function:

Remove `DEFAULT_MODEL = "gpt-4.1-mini"` line. In the endpoint signature, change `model: str = Form(default=DEFAULT_MODEL)` to `model: str | None = Form(default=None)`.

At the top of the endpoint handler, after auth check:

```python
    provider_name = get_provider_name()
    if model is None:
        model = get_default_model(provider_name)
```

Replace the API key validation block (lines 69-79). Change from checking `OPENAI_API_KEY` to checking the provider-aware key:

```python
    llm_api_key_var = get_api_key_env_var(provider_name)
    llm_api_key = os.environ.get(llm_api_key_var, "")
    ticktick_access_token = os.environ.get("TICKTICK_ACCESS_TOKEN", "")
    early_errors: list[str] = []
    if not project_name:
        early_errors.append("Project name is required")
    if not llm_api_key:
        early_errors.append(f"Missing {llm_api_key_var}")
    if not dry_run and not ticktick_access_token:
        early_errors.append("Missing TickTick access token")
    if early_errors:
        raise HTTPException(status_code=400, detail=early_errors)
```

Replace the except clause (line 134):

```python
    except (TickTickError, ProviderError, RuntimeError, requests.RequestException) as exc:
```

**Step 4: Run tests**

Run: `pytest tests/test_api_server.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add api_server.py tests/test_api_server.py
git commit -m "feat: wire providers into api_server with provider-aware key validation"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All PASS

**Step 2: Verify backward compat — default provider is openai**

Run: `python -c "from providers import get_provider_name, get_default_model; print(get_provider_name(), get_default_model(get_provider_name()))"`
Expected: `openai gpt-4.1-mini`

**Step 3: Verify anthropic selection**

Run: `LLM_PROVIDER=anthropic python -c "from providers import get_provider_name, get_default_model; print(get_provider_name(), get_default_model(get_provider_name()))"`
Expected: `anthropic claude-sonnet-4-20250514`

**Step 4: Commit any final fixups if needed, then final commit**

```bash
git add -A
git commit -m "feat: multi-provider LLM support (OpenAI + Anthropic)"
```
