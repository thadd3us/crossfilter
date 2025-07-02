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
6. **THAD TODOs**: If you see any lines marked with `THAD:`, please address the todo-item mentioned on the rest of the line!


### Code Style Guidelines:
1. **No __init__.py Files**: Avoid creating `__init__.py` files - they are unnecessary boilerplate
2. **Use Absolute Imports**: Always use absolute imports (e.g., `from crossfilter.core.session_state import SessionState`) instead of relative imports
3. **Avoid Union Types**: Don't use `Union[str, Path]` or similar - pick one type and stick with it. Union types add complexity and additional code paths. Use the most appropriate single type (e.g., `Path` for file paths)
4. **Use df_id for Row Tracking**: Use Pandas int64 index (called `df_id`) rather than UUIDs to track and refer to which points are left after filtering. These should stay stable while the app is running.
5. **Schema Column References**: When referring to a column named in a schema, don't use naked strings. Use Python identifiers, ideally derived from the Pandera schema if possible, or a parallel enum if not. Stick to the same identifier everywhere in the code (e.g. "GPS_LATITUDE") until another API forces you to change ("lat").
6. **Avoid Default Values**: Lean away from supplying default values, just make the callers be explicit (e.g., don't use `max_groups: int = 100000`).
7. **Use Python Logging**: Use Python logging rather than prints for debugging and informational output.
8. **Define Enums for Constants**: Define an enum for string constants rather than using naked strings (e.g., operation types).
9. **Prefer Stateless Functions**: Prefer stateless functions defined outside of classes when possible. Don't use classes with static methods just to scope things.
10. **Keep Logic in Python**: Keep as much logic in Python as possible, and not in Javascript. The frontend should be primarily for display and user interaction.
11. **Comprehensive Testing**: For each Python file, there should be a corresponding "*_test.py" file that concisely exercises as much functionality as possible.
12. **No Manual Testing**: Never do any sort of "ad hoc" running the server and "trying things out". Always build functionality into automated tests that can be repeated.
13. **No Global Variables**: Never create global variables. Use dependency injection or other patterns to pass state where needed.
14. **Complete Type Annotations**: Always add type annotations to all function arguments and return types.
15. **Use pytest Best Practices**: 
    - Use `tmp_path` fixture instead of `tempfile.NamedTemporaryFile`
    - Use `syrupy` plugin for testing data content in a diff-friendly way
    - Always specify return type annotations for test functions
16. **Proper Exception Handling**: Follow linter suggestions for proper exception re-raising patterns.
17. **Avoid YAGNI and Weakly Typed Fields**: Don't add fields like `metadata: Dict[str, Any]` unless there's a specific, immediate need. Prefer strongly typed, specific fields.
18. **Minimize Branching**: Avoid unnecessary `if/else` branches whenever possible. Use functions that handle edge cases (like empty collections) gracefully rather than checking for them explicitly.
19. **Enhanced Logging**: Use f-string syntax with `=` for logging (e.g., `f"{variable=}"`) to provide better context and show empty strings clearly.
20. **Start with the Simplest Solution**: Always begin with the most straightforward approach that meets the requirements. Don't copy complex patterns when simple ones will suffice.
21. **Question Infrastructure Needs**: Before adding servers, databases, or complex frameworks, ask "Is this actually needed?" For example, testing a static HTML file doesn't require a web server - use `file://` URLs instead.
22. **Prefer Synchronous over Asynchronous**: Use async/await only when truly needed for concurrency. Simple, synchronous code is easier to understand and debug.
23. **Don't Cargo-Cult Code**: Just because existing code is complex doesn't mean new code needs to be. Evaluate each situation independently and choose the appropriate level of complexity.
24. **Focus on Actual Requirements**: Implement exactly what's needed, not what might be needed. A "hello world" test should be simple, not a copy of a full application test suite.
45. **Don't enumerate arguments in docstrings**: Just use good names and type hints to make it clear what things are.

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

### Frontend Testing
- **Browser Automation**: Playwright for end-to-end testing
- **Headless Support**: Built-in headless browser support for CI/CD environments
- **Test Location**: End-to-end tests in `tests/test_frontend_e2e.py`
- **Fixtures**: Automated server startup/teardown and browser management
- **Test Markers**: Use `@pytest.mark.e2e` for frontend tests

#### Running Frontend Tests
```bash
# Install Playwright browsers (first time only)
uv run --extra dev playwright install

# Run all tests including frontend
uv run --extra dev pytest

# Run only frontend tests
uv run --extra dev pytest -m e2e

# Run frontend tests with visible browser (for debugging)
uv run --extra dev pytest -m e2e --headed

# Skip frontend tests
uv run --extra dev pytest -m "not e2e"
```

#### Headless Browser Support
Frontend tests use Playwright's built-in headless browser support, making them suitable for:
- **CI/CD Pipelines**: No need to install browsers on build servers
- **Docker Containers**: Works in headless environments
- **Development Machines**: Developers don't need browsers installed
- **Automated Testing**: Faster execution without GUI overhead

**System Dependencies Required:**
For browser tests to run, the system needs certain dependencies. Install them with:
```bash
# On Ubuntu/Debian systems
sudo playwright install-deps

# Or manually install specific packages
sudo apt-get install libnspr4 libnss3 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libxkbcommon0 libatspi2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
```

**Graceful Degradation:**
- Tests that require browsers are automatically skipped if dependencies are missing
- Server-only tests (HTTP API testing) work without browser dependencies
- Non-browser tests provide valuable coverage of the backend functionality

The test framework automatically:
- Starts the FastAPI server on a test port (8001)
- Launches headless Chromium browser (if dependencies available)
- Manages browser context and page lifecycle
- Cleans up resources after tests complete
- Skips browser tests gracefully when dependencies are missing

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