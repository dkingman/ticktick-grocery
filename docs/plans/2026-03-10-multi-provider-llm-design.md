# Multi-Provider LLM Support

## Goal

Add Anthropic Claude as an alternative vision provider alongside OpenAI, selectable via `LLM_PROVIDER` env var. Keep `--model` flag for model override within a provider. Default to OpenAI for backward compatibility.

## Architecture

### New file: `providers.py`

Provider abstraction with two implementations:

```
ProviderError(Exception)          # unified error type
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-20250514",
}
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

extract_with_openai(image_b64, mime_type, prompt, model) -> str
extract_with_anthropic(image_b64, mime_type, prompt, model) -> str
get_provider(name) -> callable     # returns one of the above
get_provider_name() -> str         # reads LLM_PROVIDER env, defaults "openai"
get_api_key_env_var(provider) -> str
get_default_model(provider) -> str
```

Each provider function:
- Instantiates its own client from the appropriate env var
- Formats the API call per provider's SDK conventions
- Returns raw text output
- Wraps provider-specific errors in `ProviderError`

### Changes to existing files

**`grocery_to_ticktick.py`:**
- `extract_ingredients()` reads provider via `providers.get_provider_name()`, delegates to `providers.get_provider(name)(...)`
- Remove direct `OpenAI` import and client instantiation
- `--model` default uses `providers.get_default_model(provider)`
- Catch `ProviderError` instead of `OpenAIError` in `main()`

**`api_server.py`:**
- `DEFAULT_MODEL` replaced with `providers.get_default_model(providers.get_provider_name())`
- API key validation uses `providers.get_api_key_env_var(provider)`
- Catch `ProviderError` instead of `OpenAIError`

**`image_utils.py`:**
- Rename `normalize_image_for_openai` -> `normalize_image`

**`validation.py`:**
- Rename `openai_api_key` field to `llm_api_key`
- Update validation message to be provider-agnostic

**`pyproject.toml`:**
- Add `anthropic>=0.40.0` dependency

## Provider-specific API details

### OpenAI (existing)
```python
client = OpenAI(api_key=...)
response = client.responses.create(
    model=model,
    input=[{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
    ]}],
)
return response.output_text
```

### Anthropic (new)
```python
client = Anthropic(api_key=...)
message = client.messages.create(
    model=model,
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
        {"type": "text", "text": prompt},
    ]}],
)
return message.content[0].text
```

## Testing plan

### New: `tests/test_providers.py`
- `test_get_provider_returns_openai_by_default`
- `test_get_provider_returns_anthropic_when_set`
- `test_get_provider_raises_on_unknown_provider`
- `test_get_default_model_openai`
- `test_get_default_model_anthropic`
- `test_get_api_key_env_var_openai`
- `test_get_api_key_env_var_anthropic`
- `test_extract_with_openai_calls_client_correctly` (mock OpenAI client, verify call shape)
- `test_extract_with_openai_wraps_error_in_provider_error`
- `test_extract_with_anthropic_calls_client_correctly` (mock Anthropic client, verify call shape)
- `test_extract_with_anthropic_wraps_error_in_provider_error`
- `test_extract_with_openai_reads_api_key_from_env`
- `test_extract_with_anthropic_reads_api_key_from_env`
- `test_missing_api_key_raises_provider_error`

### Updated: `tests/test_grocery_to_ticktick.py`
- Update `ExtractIngredientsTests` to mock `providers` instead of `OpenAI` directly
- Update `MainErrorHandlingTests` to use `ProviderError` instead of `OpenAIError`
- Update HEIC test to reference `normalize_image` instead of `normalize_image_for_openai`

### Updated: `tests/test_api_server.py`
- Update `base_env` to be provider-aware (test both providers)
- Update upstream failure test to use `ProviderError`
- Add test for Anthropic provider via env var

### Updated: `tests/test_validation.py`
- Rename `openai_api_key` references to `llm_api_key`
- Update error message assertions

## Env var summary

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `"openai"` (default) or `"anthropic"` |
| `OPENAI_API_KEY` | Required when provider is openai |
| `ANTHROPIC_API_KEY` | Required when provider is anthropic |

## Backward compatibility

- `LLM_PROVIDER` defaults to `"openai"` — no change needed for existing setups
- `--model` still defaults to `gpt-4.1-mini` when provider is openai
- `OPENAI_API_KEY` still works as before
