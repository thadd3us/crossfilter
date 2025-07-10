FROM python:3.11

USER root

RUN apt-get update && apt-get install -y \
    git \
    ncdu

# Install uv.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Make the dev user.
RUN adduser --disabled-password --comment 'dev user' dev

USER dev

# Some general things I like.
RUN curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh \
    curl -sS https://starship.rs/install.sh | sh


# What do we need for this project??
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./

# Pre-heat uv.
RUN uv sync --extra=dev

# Install playwright system deps -- must be root.
# THIS TAKES ~2 minutes.
USER root
RUN .venv/bin/playwright install-deps
USER dev

# Pre-heat playwright.
RUN .venv/bin/playwright install
