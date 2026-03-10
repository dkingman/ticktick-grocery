# grocery-to-ticktick

Small CLI to:
1. Read an image of a recipe / handwritten note
2. Extract grocery ingredients
3. Add them as tasks to a TickTick list you pass via `--project`
4. Run TickTick OAuth automatically when no access token is present

Also includes an HTTP API service:
- `POST /api/ticktick/import` for image upload + TickTick import
- Bearer auth via `Authorization: Bearer <API_KEY>`

## Prerequisites

- `uv`
- Python 3.10+ (managed by `uv`)
- TickTick OAuth app (`client_id` + `client_secret`)
- HEIC/HEIF support requires `libheif` (Docker image installs `libheif1`).

## One-command run

Set your TickTick app redirect URI to:

```text
http://127.0.0.1:8765/callback
```

Then run a single command:

```bash
uv run grocery_to_ticktick.py /path/to/image.jpg \
  --project "Errands" \
  --ticktick-client-id "YOUR_CLIENT_ID" \
  --ticktick-client-secret "YOUR_CLIENT_SECRET" \
  --oauth-open-browser
```

If `TICKTICK_ACCESS_TOKEN` is not set, the script handles OAuth, then continues to import grocery items.

## Optional env vars

You can avoid passing args each time:

```bash
export TICKTICK_CLIENT_ID="..."
export TICKTICK_CLIENT_SECRET="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."   # only needed when using --provider anthropic
```

With these set:

```bash
uv run grocery_to_ticktick.py /path/to/image.jpg --project "Errands" --oauth-open-browser
```

## Reuse access token

After first run, if you already have an access token:

```bash
export TICKTICK_ACCESS_TOKEN="..."
```

Then run without OAuth:

```bash
uv run grocery_to_ticktick.py /path/to/image.jpg --project "Errands"
```

## Standalone token helper (optional)

Set your TickTick app redirect URI to:

```text
http://127.0.0.1:8765/callback
```

Run:

```bash
uv run ticktick_oauth_helper.py \
  --client-id "YOUR_CLIENT_ID" \
  --client-secret "YOUR_CLIENT_SECRET" \
  --open-browser
```

After authorization, it prints:

```bash
export TICKTICK_ACCESS_TOKEN="..."
export TICKTICK_REFRESH_TOKEN="..."
```

Options:

- `--project "Errands"` required target list name
- `--dry-run` to print extracted ingredients only (no TickTick auth/API calls)
- `--provider openai|anthropic` to switch LLM provider (default: `openai`)
- `--model ...` to choose another vision-capable model (defaults to `gpt-4.1-mini` for OpenAI, `claude-3-5-sonnet-latest` for Anthropic)
- `--oauth-open-browser` to open TickTick auth URL automatically

## TickTick API notes

This script uses TickTick Open API endpoints:

- `GET /open/v1/project`
- `GET /open/v1/project/{projectId}/data`
- `POST /open/v1/task`

Base URL: `https://api.ticktick.com/open/v1`

To get a bearer token, create an OAuth app in TickTick developer console and exchange an auth code for an access token. Then set `TICKTICK_ACCESS_TOKEN`.

## HTTP API server

The API server is separate from the CLI OAuth flow and uses a static
`TICKTICK_ACCESS_TOKEN`.

Required environment variables:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."   # required if provider=anthropic
export TICKTICK_ACCESS_TOKEN="..."
export API_KEY="..."
```

Optional environment variables:

```bash
export DEFAULT_TICKTICK_PROJECT="Errands"   # used when request omits project
export DEFAULT_MODEL_PROVIDER="openai"      # optional: openai|anthropic
export MAX_UPLOAD_BYTES="52428800"          # default 50 MiB
```

Run locally:

```bash
uv run uvicorn api_server:app --host 127.0.0.1 --port 8090
```

For development, add `--reload` to auto-restart on code changes:

```bash
uv run uvicorn api_server:app --host 127.0.0.1 --port 8090 --reload
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8090/api/ticktick/import" \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@/path/to/image.jpg" \
  -F "provider=openai" \
  -F "project=Errands" \
  -F "dry_run=false"
```

HEIC/HEIF uploads are automatically converted to JPEG before sending to OpenAI.

Success response:

```json
{
  "project": "Errands",
  "ingredients": ["milk", "bread"],
  "created": ["milk", "bread"],
  "skipped": [],
  "dry_run": false
}
```

Error status codes:
- `400` bad request (missing image/project)
- `401` unauthorized
- `413` uploaded file too large
- `415` unsupported media type
- `502` upstream OpenAI/TickTick failures
- `500` server misconfiguration/internal error

## Container image

Build locally:

```bash
docker build -t ticktick-grocery-api:local .
```

Run locally:

```bash
docker run --rm -p 8090:8090 \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -e TICKTICK_ACCESS_TOKEN \
  -e API_KEY \
  -e DEFAULT_TICKTICK_PROJECT \
  ticktick-grocery-api:local
```

## GHCR publish workflow

`/.github/workflows/publish-image.yml` builds and pushes the image to GHCR on
every push to `main` and on manual dispatch. It emits a digest-pinned image ref:

`ghcr.io/<owner>/<repo>@sha256:<digest>`
