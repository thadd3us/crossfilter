FROM python:3.11 AS build_stage_1

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

# Sculptor compatibility for now.
WORKDIR /user_home/workspace

# Sculptor compatibility.
# This should get the uv, playwright, and HF caches to materialize here.
ENV HOME=/user_home

# What do we need for this project??
# TODO: Copy "fake" frozen versions of these to the container, so it doesn't always need to be rebuilt if they change.
COPY pyproject.toml uv.lock README.md ./

# Pre-heat uv.
RUN uv sync --extra=dev

# Install playwright system deps -- must be root, but has to happen after we have uv and the pyproject.toml installed.
# THIS TAKES ~2 minutes.
USER root
RUN .venv/bin/playwright install-deps


FROM build_stage_1 AS prepare_user_home

USER dev

# Copy warm cache scripts to /tmp and run them.
COPY dev/warm_cache/01_playwright_install.py /tmp/warm_cache/01_playwright_install.py
COPY dev/warm_cache/02_hf_model_download.py /tmp/warm_cache/02_hf_model_download.py

# Pre-heat playwright and HF using warm cache script.
RUN .venv/bin/python /tmp/warm_cache/01_playwright_install.py
RUN .venv/bin/python /tmp/warm_cache/02_hf_model_download.py
# Clean up warm cache scripts.

FROM build_stage_1 AS copy_user_home

# Sculptor compatibility for now; we're going to be user501 in Sculptor and need to work in this directory.
COPY --from=prepare_user_home --chown=dev:dev --chmod=a+rwX /user_home /user_home

USER dev
WORKDIR /user_home/
