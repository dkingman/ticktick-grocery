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


if __name__ == "__main__":
    unittest.main()
