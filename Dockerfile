FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8090
ENV MAX_UPLOAD_BYTES=52428800

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN apt-get update \
  && apt-get install -y --no-install-recommends libheif1 \
  && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 appuser
RUN uv sync --frozen --no-dev
RUN chown -R appuser:appuser /app/.venv

COPY . .

USER appuser

EXPOSE 8090

CMD ["uv", "run", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8090"]
