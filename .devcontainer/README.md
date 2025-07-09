# GitHub Codespaces Development

## Quick Start

1. Create a Codespace from your repository
2. Wait for setup to complete
3. Start developing!

## Commands

```bash
# Start the server
uv run python -m crossfilter.main serve

# Run tests
uv run --extra dev pytest

# Run frontend tests
uv run --extra dev pytest -m e2e
```