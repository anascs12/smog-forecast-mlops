# ---- build stage: resolve dependencies and bake in the verified model ----
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
# Download and SHA-256-verify the model at build time, so the image is self-contained
# and a container never starts with an unverified artifact.
ENV MODEL_CACHE_DIR=/app/model_cache
RUN /app/.venv/bin/python -c "from smog_service import registry; print(registry.load().card.version)"

# ---- runtime stage: slim image, non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/model_cache /app/model_cache
ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_CACHE_DIR=/app/model_cache \
    PREDICTION_DB=/tmp/predictions.db \
    PORT=7860 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "uvicorn smog_service.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
