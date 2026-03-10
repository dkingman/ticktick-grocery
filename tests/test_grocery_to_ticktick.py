import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from providers import ProviderError

import grocery_to_ticktick as app


class ParseModelJsonTests(unittest.TestCase):
    def test_parses_plain_json_object(self) -> None:
        payload = app._parse_json_from_model_text('{"ingredients":["milk"]}')
        self.assertEqual(payload, {"ingredients": ["milk"]})

    def test_parses_fenced_json_object(self) -> None:
        payload = app._parse_json_from_model_text(
            '```json\n{"ingredients":["milk","bread"]}\n```'
        )
        self.assertEqual(payload, {"ingredients": ["milk", "bread"]})

    def test_extracts_first_valid_object_from_mixed_text(self) -> None:
        text = 'prefix {"ingredients":["milk"]} suffix {oops}'
        payload = app._parse_json_from_model_text(text)
        self.assertEqual(payload, {"ingredients": ["milk"]})

    def test_raises_runtime_error_when_no_valid_json_object(self) -> None:
        with self.assertRaises(RuntimeError):
            app._parse_json_from_model_text("not json at all")


class ParseArgsTests(unittest.TestCase):
    def test_project_is_required(self) -> None:
        with patch.object(sys, "argv", ["grocery_to_ticktick.py", "/tmp/image.jpg"]):
            with self.assertRaises(SystemExit) as exc:
                app.parse_args()
        self.assertEqual(exc.exception.code, 2)


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

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
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


class SyncItemsTests(unittest.TestCase):
    def test_sync_items_to_project_tracks_created_and_skipped(self) -> None:
        with patch("grocery_to_ticktick.get_project_id", return_value="project-123"):
            with patch(
                "grocery_to_ticktick.get_existing_task_titles",
                return_value={"milk"},
            ):
                with patch("grocery_to_ticktick.create_task") as mock_create:
                    created, skipped = app.sync_items_to_project(
                        "token", "Errands", ["Milk", "Bread", "Bread"]
                    )

        self.assertEqual(created, ["Bread"])
        self.assertEqual(skipped, ["Milk", "Bread"])
        mock_create.assert_called_once_with("token", "Bread", "project-123")


if __name__ == "__main__":
    unittest.main()
