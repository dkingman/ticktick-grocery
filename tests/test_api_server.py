import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai import OpenAIError

import api_server


class TickTickApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_server.app)
        self.base_env = {
            "API_KEY": "secret-key",
            "TICKTICK_ACCESS_TOKEN": "ticktick-token",
            "OPENAI_API_KEY": "openai-key",
            "DEFAULT_TICKTICK_PROJECT": "Default Errands",
            "MAX_UPLOAD_BYTES": "52428800",
        }

    def _auth_headers(self, token: str = "secret-key") -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_missing_auth_returns_401(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            response = self.client.post(
                "/api/ticktick/import",
                files={"image": ("list.png", b"fake", "image/png")},
            )
        self.assertEqual(response.status_code, 401)

    def test_invalid_auth_returns_401(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            response = self.client.post(
                "/api/ticktick/import",
                headers=self._auth_headers(token="wrong"),
                files={"image": ("list.png", b"fake", "image/png")},
            )
        self.assertEqual(response.status_code, 401)

    def test_missing_image_returns_400(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            response = self.client.post(
                "/api/ticktick/import",
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 400)

    def test_non_image_upload_returns_415(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            response = self.client.post(
                "/api/ticktick/import",
                headers=self._auth_headers(),
                files={"image": ("note.txt", b"not-image", "text/plain")},
            )
        self.assertEqual(response.status_code, 415)

    def test_heic_upload_without_suffix_uses_heic_extension(self) -> None:
        captured: dict = {}

        def fake_extract(path, model):
            captured["path"] = path
            return []

        with patch.dict(os.environ, self.base_env, clear=True):
            with patch("api_server.extract_ingredients", side_effect=fake_extract):
                response = self.client.post(
                    "/api/ticktick/import",
                    headers=self._auth_headers(),
                    files={"image": ("upload", b"fake", "image/heic")},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["path"].suffix, ".heic")

    def test_oversized_upload_returns_413(self) -> None:
        env = dict(self.base_env)
        env["MAX_UPLOAD_BYTES"] = "3"
        with patch.dict(os.environ, env, clear=True):
            with patch("api_server.extract_ingredients") as mock_extract:
                response = self.client.post(
                    "/api/ticktick/import",
                    headers=self._auth_headers(),
                    files={"image": ("list.png", b"1234", "image/png")},
                )
        self.assertEqual(response.status_code, 413)
        mock_extract.assert_not_called()

    def test_dry_run_uses_default_project_and_skips_creation(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            with patch("api_server.extract_ingredients", return_value=["Milk", "Bread"]):
                with patch("api_server.sync_items_to_project") as mock_sync:
                    response = self.client.post(
                        "/api/ticktick/import",
                        headers=self._auth_headers(),
                        files={"image": ("list.png", b"fake", "image/png")},
                        data={"dry_run": "true"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "project": "Default Errands",
                "ingredients": ["Milk", "Bread"],
                "created": [],
                "skipped": [],
                "dry_run": True,
            },
        )
        mock_sync.assert_not_called()

    def test_project_override_and_sync_result(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            with patch("api_server.extract_ingredients", return_value=["Milk", "Bread"]):
                with patch(
                    "api_server.sync_items_to_project",
                    return_value=(["Bread"], ["Milk"]),
                ) as mock_sync:
                    response = self.client.post(
                        "/api/ticktick/import",
                        headers=self._auth_headers(),
                        files={"image": ("list.png", b"fake", "image/png")},
                        data={"project": "Errands"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "project": "Errands",
                "ingredients": ["Milk", "Bread"],
                "created": ["Bread"],
                "skipped": ["Milk"],
                "dry_run": False,
            },
        )
        mock_sync.assert_called_once_with(
            "ticktick-token", "Errands", ["Milk", "Bread"]
        )

    def test_upstream_failures_map_to_502(self) -> None:
        with patch.dict(os.environ, self.base_env, clear=True):
            with patch(
                "api_server.extract_ingredients", side_effect=OpenAIError("boom")
            ):
                response = self.client.post(
                    "/api/ticktick/import",
                    headers=self._auth_headers(),
                    files={"image": ("list.png", b"fake", "image/png")},
                )
        self.assertEqual(response.status_code, 502)
        self.assertIn("boom", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
