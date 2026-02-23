FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8090
ENV MAX_UPLOAD_BYTES=52428800

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8090

CMD ["uv", "run", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8090"]
