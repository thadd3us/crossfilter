# GitHub Codespaces Development Environment

This directory contains the configuration for developing Crossfilter in GitHub Codespaces.

## Quick Start

1. **Create a Codespace**: Click the "Code" button in the GitHub repository and select "Create codespace on main" (or your current branch)

2. **Wait for setup**: The container will automatically:
   - Install Python dependencies using `uv`
   - Install Playwright browsers for frontend testing
   - Configure VS Code extensions and settings
   - Forward ports 8000 and 8001

3. **Start developing**: Once the setup is complete, you can:
   - Run the server: `uv run python -m crossfilter.main serve --host 0.0.0.0`
   - Run tests: `uv run --extra dev pytest`
   - Run frontend tests: `uv run --extra dev pytest -m e2e`

## Features

### Pre-installed Tools
- **Python 3.11** with `uv` package manager
- **Git** and **GitHub CLI** for version control
- **Playwright** with browsers for frontend testing
- **All project dependencies** (including dev dependencies)

### VS Code Extensions
- Python support with debugging, linting, and formatting
- Jupyter notebook support
- Playwright test runner
- JSON editing support
- Code formatting and import organization

### Port Forwarding
- **Port 8000**: Main Crossfilter application (auto-opens in browser)
- **Port 8001**: Test server (for development)

## Development Workflow

### Running the Application
```bash
# Start the development server
uv run python -m crossfilter.main serve --host 0.0.0.0

# The app will be available at the forwarded port URL
```

### Testing
```bash
# Run all tests
uv run --extra dev pytest

# Run only backend tests
uv run --extra dev pytest -m "not e2e"

# Run only frontend tests
uv run --extra dev pytest -m e2e

# Run tests with coverage
uv run --extra dev pytest --cov=crossfilter
```

### Code Quality
```bash
# Format code
uv run --extra dev black .

# Lint code
uv run --extra dev ruff check .

# Type checking
uv run --extra dev mypy crossfilter/
```

## Container Configuration

### Environment Variables
- `PYTHONPATH`: Set to `/workspace` for proper module imports
- `PATH`: Includes the virtual environment's bin directory

### Security
- Container runs as `vscode` user for security
- Limited privileges with necessary capabilities for debugging
- Safe directory configuration for Git

### Performance
- Optimized Docker build with multi-stage approach
- Cached dependency installation
- Efficient file watching and hot reload

## Troubleshooting

### Common Issues

1. **Port forwarding not working**: 
   - Check if the server is bound to `0.0.0.0` instead of `localhost`
   - Verify the port is properly forwarded in the Codespace

2. **Playwright tests failing**:
   - Ensure browsers are installed: `uv run --extra dev playwright install`
   - Check system dependencies are available

3. **Python import errors**:
   - Verify `PYTHONPATH` is set correctly
   - Check if virtual environment is activated

4. **Performance issues**:
   - Consider using a larger Codespace machine type
   - Check if too many processes are running

### Manual Setup Commands

If automatic setup fails, you can run these commands manually:

```bash
# Install dependencies
uv sync --extra dev

# Install Playwright browsers
uv run --extra dev playwright install

# Set up Git safe directory
git config --global --add safe.directory /workspace
```

## Customization

### Adding VS Code Extensions
Edit `.devcontainer/devcontainer.json` and add extension IDs to the `extensions` array.

### Modifying Port Forwarding
Update the `forwardPorts` array in `devcontainer.json` to add or remove ports.

### Environment Variables
Add or modify environment variables in the `containerEnv` section.

### System Dependencies
Add system packages to the `apt-get install` command in the Dockerfile.

## Local Development

This devcontainer configuration can also be used for local development with:
- **VS Code Dev Containers extension**
- **Docker Desktop**
- **GitHub CLI** (for Codespaces-like experience)

To use locally:
1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Run "Dev Containers: Reopen in Container" from the command palette