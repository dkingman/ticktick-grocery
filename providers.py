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
