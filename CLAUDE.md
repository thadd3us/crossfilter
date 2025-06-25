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
6. **ALWAYS Run Full Test Suite**: Before claiming any task is complete, you MUST run `uv run --extra dev pytest` to ensure all tests pass

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