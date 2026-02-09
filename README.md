# grocery-to-ticktick

Small CLI to:
1. Read an image of a recipe / handwritten note
2. Extract grocery ingredients
3. Add them as tasks to a TickTick list you pass via `--project`
4. Run TickTick OAuth automatically when no access token is present

## Prerequisites

- `uv`
- Python 3.10+ (managed by `uv`)
- TickTick OAuth app (`client_id` + `client_secret`)

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
- `--model gpt-4.1-mini` to choose another vision-capable model
- `--oauth-open-browser` to open TickTick auth URL automatically

## TickTick API notes

This script uses TickTick Open API endpoints:

- `GET /open/v1/project`
- `GET /open/v1/project/{projectId}/data`
- `POST /open/v1/task`

Base URL: `https://api.ticktick.com/open/v1`

To get a bearer token, create an OAuth app in TickTick developer console and exchange an auth code for an access token. Then set `TICKTICK_ACCESS_TOKEN`.
