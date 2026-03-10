import unittest
from pathlib import Path

from validation import ImportRequest, validate_import_request


class ValidateImportRequestTests(unittest.TestCase):
    def _valid_request(self, **overrides) -> ImportRequest:
        """Build a valid ImportRequest, allowing field overrides."""
        defaults = {
            "image_path": Path(__file__),  # any existing file
            "project": "Groceries",
            "provider": "openai",
            "openai_api_key": "sk-test",
            "anthropic_api_key": "anthropic-test",
            "ticktick_access_token": "tt-token",
        }
        defaults.update(overrides)
        return ImportRequest(**defaults)

    def test_valid_request_passes(self) -> None:
        req = self._valid_request()
        self.assertEqual(validate_import_request(req), [])

    def test_missing_project_returns_error(self) -> None:
        req = self._valid_request(project="")
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("Project", errors[0])

    def test_whitespace_only_project_returns_error(self) -> None:
        req = self._valid_request(project="   ")
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("Project", errors[0])

    def test_missing_openai_api_key_returns_error(self) -> None:
        req = self._valid_request(openai_api_key="")
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("OPENAI_API_KEY", errors[0])

    def test_missing_anthropic_api_key_returns_error(self) -> None:
        req = self._valid_request(provider="anthropic", anthropic_api_key="")
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("ANTHROPIC_API_KEY", errors[0])

    def test_missing_ticktick_token_when_not_dry_run(self) -> None:
        req = self._valid_request(ticktick_access_token="", dry_run=False)
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("TickTick", errors[0])

    def test_dry_run_allows_missing_ticktick_token(self) -> None:
        req = self._valid_request(ticktick_access_token="", dry_run=True)
        self.assertEqual(validate_import_request(req), [])

    def test_nonexistent_image_path_returns_error(self) -> None:
        req = self._valid_request(image_path=Path("/nonexistent/image.png"))
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 1)
        self.assertIn("Image not found", errors[0])

    def test_multiple_errors_accumulate(self) -> None:
        req = ImportRequest(
            image_path=Path("/nonexistent/image.png"),
            project="",
            openai_api_key="",
            ticktick_access_token="",
        )
        errors = validate_import_request(req)
        self.assertEqual(len(errors), 4)


if __name__ == "__main__":
    unittest.main()
