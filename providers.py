"""Pluggable LLM provider abstraction for vision extraction."""

import logging
import os

from anthropic import Anthropic, APIError
from openai import OpenAI, OpenAIError

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


def extract_with_anthropic(
    image_b64: str, mime_type: str, prompt: str, model: str
) -> str:
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


_PROVIDERS = {
    "openai": extract_with_openai,
    "anthropic": extract_with_anthropic,
}


def get_provider(name: str):
    if name not in _PROVIDERS:
        raise ProviderError(f"Unknown provider: {name}")
    return _PROVIDERS[name]
