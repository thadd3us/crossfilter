FROM python:3.11

RUN pip install uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra dev

RUN uv run --extra dev playwright install-deps
RUN uv run --extra dev playwright install

COPY . .