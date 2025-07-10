FROM python:3.11

USER root

# Make the dev user.
RUN adduser --disabled-password dev

RUN apt-get update && apt-get install -y \
    direnv \
    build-essential \
    curl \
    git \
    ncdu

# Some general things I like.
RUN \
    curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh && \
    curl -sS https://starship.rs/install.sh | sh -s -- --yes && \
    echo "Done."

USER dev


RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/home/dev/uv_install/" sh
ENV PATH="/home/dev/uv_install:$PATH"
RUN uv --version

# What do we need for this project??
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./

# Pre-heat uv.
RUN uv sync --extra=dev

# Install playwright system deps -- must be root, but has to happen after we have uv and the pyproject.toml installed.
# THIS TAKES ~2 minutes.
USER root
RUN .venv/bin/playwright install-deps
USER dev

# Pre-heat playwright.
RUN .venv/bin/playwright install
