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
