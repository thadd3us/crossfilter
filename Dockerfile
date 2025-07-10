FROM python:3.11

USER root

# Install uv.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Make the dev user.
RUN adduser --disabled-password --comment 'dev user' dev

# What do we need?
USER dev
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./

# Pre-heat uv.
RUN uv sync --extra=dev

# Install playwright system deps -- must be root.
USER root
RUN .venv/bin/playwright install-deps
USER dev

# Pre-heat playwright.
RUN .venv/bin/playwright install
