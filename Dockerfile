FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra dev

RUN uv run --extra dev playwright install-deps
RUN uv run --extra dev playwright install

COPY . .