# Crossfilter - Design Summary for Claude

## Project Overview

Crossfilter is a Python web application for interactive crossfiltering and analysis of geospatial and temporal data, with a focus on large collections of GPX files and other datasets. It uses a single-session architecture inspired by applications like Jupyter Notebook, ParaView, and VisIt.

## Architecture Pattern: Single-Session Web Application

### Design Philosophy
- **One Session Per Server Instance**: Each web server instance maintains exactly one active session state
- **Browser as UI**: Uses modern web technologies for rich, interactive user interface
- **Local Application**: Designed to run locally like a desktop app, but with web UI flexibility
- **No Multi-User Complexity**: Eliminates need for session management, authentication, or user coordination

### Key Components

#### 1. SessionState Class (`crossfilter/core/session_state.py`)
- **Purpose**: Manages the current state of data loaded into the application
- **Contains**: Pandas DataFrame for main dataset, metadata dictionary
- **Location**: Global singleton instance in `main.py` as `session_state`
- **Features**:
  - Data loading and clearing
  - Metadata tracking (shape, columns, dtypes, memory usage)
  - Status summaries
  - Future-ready for Pandera schema validation

#### 2. FastAPI Web Server (`crossfilter/main.py`)
- **Framework**: FastAPI for REST API and web serving
- **CLI**: Typer for command-line interface
- **Endpoints**:
  - `/`: Main HTML interface
  - `/api/session`: Session state status endpoint
- **Server**: Uvicorn for ASGI serving (default: localhost:8000)

#### 3. Data Processing Pipeline (Planned)
- **GPX Parsing**: Extract trackpoints with coordinates, timestamps, metadata
- **Spatial Indexing**: H3 hexagonal indexing for efficient spatial queries
- **Temporal Grouping**: Minute/hour/day/month aggregation
- **Cross-filtering**: Bidirectional selection between spatial and temporal views

## Development Guidelines

### When Working on This Codebase:
1. **Maintain Single-Session Pattern**: All state should go through the global `session_state` instance
2. **Follow FastAPI Patterns**: Use async/await, proper response models, dependency injection
3. **Preserve Performance Focus**: Remember this is designed for large datasets (millions of points)
4. **Use Existing Dependencies**: Pandas, NumPy, H3, Plotly, FastAPI, Typer, Pydantic
5. **Test Commands**: `uv run --extra dev pytest` for testing

### Code Style Guidelines:
1. **No __init__.py Files**: Avoid creating `__init__.py` files - they are unnecessary boilerplate
2. **Use Absolute Imports**: Always use absolute imports (e.g., `from crossfilter.core.session_state import SessionState`) instead of relative imports

### Project Structure
```
crossfilter/
├── core/           # Core data processing and state management
│   └── session_state.py
├── visualization/  # Plotting and rendering components (future)
├── web/           # FastAPI routes and web components (future)  
├── static/        # Frontend assets (HTML, CSS, JS) (future)
├── __main__.py    # Entry point
└── main.py        # Main application and CLI
```

### Similar Applications (Prior Art)
- **Jupyter Notebook**: Local server (localhost:8888) with browser interface for data analysis
- **ParaViewWeb**: Web-based access to ParaView's scientific visualization capabilities
- **VisIt**: Interactive parallel visualization with local client-server support

### Testing and Development
- **Test Framework**: pytest with coverage
- **Code Quality**: Black (formatting), Ruff (linting), MyPy (type checking)
- **Package Manager**: uv for dependency management
- **Python Version**: 3.9+

## Current Implementation Status

✅ **Completed**:
- Basic FastAPI application structure
- CLI interface with Typer
- SessionState class with Pandas DataFrame support
- Global session state instance
- Basic API endpoint for session status
- Documentation updates

🔄 **Next Steps**:
- GPX file parsing and loading
- Spatial indexing with H3
- Interactive visualization components
- Cross-filtering logic
- Frontend interface development
- Pandera schema validation

## Usage

```bash
# Start development server
uv run python -m crossfilter.main serve

# Or with custom port
uv run python -m crossfilter.main serve --port 8080

# Run tests
uv run --extra dev pytest
```

Access at: http://localhost:8000