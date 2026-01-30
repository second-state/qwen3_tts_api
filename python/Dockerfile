FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py .

ENV QWEN_TTS_DEVICE=cpu \
    QWEN_TTS_DTYPE=float32 \
    QWEN_TTS_ATTN="" \
    HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONWARNINGS="ignore::SyntaxWarning"

EXPOSE 8000

CMD ["/app/.venv/bin/python", "main.py"]
