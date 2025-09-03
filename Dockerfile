FROM ubuntu:24.04

USER root
RUN apt-get update && apt-get install -y \
    direnv \
    build-essential \
    curl \
    git \
    ncdu \
    adduser
USER dev

# Sculptor compatibility.
# This should get the uv, playwright, and HF caches to materialize here.
# Make the dev user.
ENV HOME=/user_home
WORKDIR /user_home

USER root
RUN adduser --disabled-password --home ${HOME} dev
RUN chown -R dev:dev ${HOME} && chmod -R a+rwX ${HOME}
USER dev

RUN mkdir -p ${HOME}/bin
# Some general things I like.
RUN \
    curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh && \
    curl -sS https://starship.rs/install.sh | sh -s -- --yes --bin-dir ${HOME}/bin && \
    echo "Done."

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${HOME}/uv_install/" sh
ENV PATH="${HOME}/uv_install:$PATH"
RUN uv --version

# Sculptor compatibility for now.
WORKDIR /user_home/workspace

# What do we need for this project??
# TODO: Copy "fake" frozen versions of these to the container, so it doesn't always need to be rebuilt if they change.
COPY pyproject.toml uv.lock README.md ./

# Pre-heat uv.
RUN uv sync --extra=dev

# Install playwright system deps -- must be root, but has to happen after we have uv and the pyproject.toml installed.
# THIS TAKES ~2 minutes.
USER root
RUN /user_home/workspace/.venv/bin/playwright install-deps
USER dev

WORKDIR /tmp/warm_cache
# Copy warm cache scripts to /tmp and run them.
COPY dev/warm_cache/01_playwright_install.py /tmp/warm_cache/01_playwright_install.py
COPY dev/warm_cache/02_hf_model_download.py /tmp/warm_cache/02_hf_model_download.py

# Pre-heat playwright and HF using warm cache script.
RUN /user_home/workspace/.venv/bin/python /tmp/warm_cache/01_playwright_install.py
RUN /user_home/workspace/.venv/bin/python /tmp/warm_cache/02_hf_model_download.py

# Sculptor compatibility for now; we're going to be user501 in Sculptor and need to work in this directory.
USER root
RUN chown -R dev:dev ${HOME} && chmod -R a+rwX ${HOME}
USER dev

ENV PATH="${HOME}/.atuin/bin/:$PATH"

WORKDIR /user_home/
