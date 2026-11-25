# Nocturne API. Built and run by Render (render.yaml at the repo root); works on any host
# that runs a Docker image and sets $PORT.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_CACHE_DIR=/app/models

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml ./
RUN mkdir app scripts && touch app/__init__.py scripts/__init__.py \
    && pip install ".[embeddings]" \
    && pip uninstall -y nocturne-backend \
    && rm -rf app scripts build *.egg-info

# Bake the embedding model into the image: free hosts have no persistent disk, so it would
# otherwise download again on every cold start.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('$EMBEDDING_MODEL', cache_dir='/app/models')"; \
    fi

COPY app ./app
COPY scripts ./scripts
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 nocturne && chown -R nocturne /app
USER nocturne

EXPOSE 8000
# Apply any new migrations, then serve. Render sets $PORT.
CMD ["sh", "-c", "python -m scripts.migrate && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
