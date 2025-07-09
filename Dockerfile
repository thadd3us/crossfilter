# Dockerfile for Crossfilter development in GitHub Codespaces
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # Dependencies for Playwright browsers
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN pip install uv

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Install Python dependencies using uv
RUN uv sync --extra dev

# Install Playwright browsers
RUN uv run --extra dev playwright install

# Expose the default port for the application
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/workspace
ENV PATH="/workspace/.venv/bin:$PATH"

# Default command
CMD ["uv", "run", "python", "-m", "crossfilter.main", "serve", "--host", "0.0.0.0", "--port", "8000"]