import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from openai import OpenAIError

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

            response = type("Response", (), {"output_text": '{"ingredients":["milk"]}'})()
            captured: dict = {}

            class FakeClient:
                def __init__(self) -> None:
                    self.responses = self

                def create(self, **kwargs):
                    captured.update(kwargs)
                    return response

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
                with patch("grocery_to_ticktick.OpenAI", return_value=FakeClient()):
                    items = app.extract_ingredients(image_path, "gpt-4.1-mini")

            self.assertEqual(items, ["milk"])
            image_url = captured["input"][0]["content"][1]["image_url"]
            self.assertTrue(image_url.startswith("data:image/png;base64,"))

    def test_rejects_unsupported_non_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.txt"
            image_path.write_text("not an image", encoding="utf-8")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
                with self.assertRaises(RuntimeError):
                    app.extract_ingredients(image_path, "gpt-4.1-mini")


class MainErrorHandlingTests(unittest.TestCase):
    def test_main_returns_one_on_openai_error(self) -> None:
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
                        side_effect=OpenAIError("boom"),
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
