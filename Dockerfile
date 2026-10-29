FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_NO_PROGRESS=1 \
    EMBEDDING_CACHE_DIR=/models \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.12.15

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./README.md
COPY app ./app
COPY scripts ./scripts
COPY main.py ./main.py
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /models \
    && chown app:app /models

USER app
EXPOSE 2003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2003"]
