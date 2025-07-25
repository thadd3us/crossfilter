# Crossfilter

A Python web application for interactive crossfiltering and analysis of geospatial and temporal data, with support for large collections of GPX files and other datasets.

## Features

- **Large-scale GPX processing**: Handle hundreds to thousands of GPX files with millions of trackpoints
- **Interactive visualization**: Browser-based UI with zoomable maps and temporal plots
- **Cross-filtering**: Select points spatially on a map and temporally on a CDF plot
- **Hierarchical aggregation**: Uses H3 spatial indexing and temporal grouping for performance
- **Individual point annotation**: Support for annotating individual points when dataset is filtered
- **Real-time filtering**: Incremental spatial and temporal filtering with live updates

## Architecture

### Single-Session Design

Crossfilter uses a **single-session architecture** where each instance of the web server maintains exactly one active session state. This design is inspired by successful local web applications like Jupyter Notebook, ParaView, and VisIt, which run local web servers and use the browser as a rich UI client.

**Key Benefits:**
- **Simplified State Management**: No need for complex session management, user authentication, or multi-user coordination
- **Rich Browser UI**: Leverages modern web technologies for interactive visualizations and responsive interfaces
- **Local Performance**: Directaccess to local file system and full computational resources
- **Familiar Development Stack**: Uses standard web technologies (FastAPI, HTML/CSS/JS) while maintaining desktop-like performance

**Similar Applications:**
- Jupyter Notebook: Runs local server (typically localhost:8888) with browser-based notebook interface
- ParaViewWeb: Provides web-based access to ParaView's visualization capabilities
- VisIt: Supports local client-serve r mode for interactive scientific visualization

This pattern is ideal for data analysis applications where the primary use case is individual researchers working with their local datasets, similar to how one would use a traditional desktop application but with the flexibility and rich UI capabilities of modern web technologies.

### Data Processing & Projections
- **GPX parsing**: Extracts trackpoints with coordinates, timestamps, and metadata
- **Spatial indexing**: H3 hexagonal indexing for efficient spatial queries and aggregation
- **Temporal grouping**: Minute/hour/day/month aggregation for temporal analysis
- **Data projections**: Multiple simultaneous views of the same underlying dataset
- **Adaptive resolution**: Automatically adjusts visualization detail based on data size

### Data Projections & Visualization
- **Geographic projection**: Spatial heatmaps and point clouds with H3 aggregation
- **Temporal projection**: Cumulative distribution functions and timeline views
- **Cross-filtering**: Bidirectional selection between different data projections
- **Progressive detail**: Switch between aggregated and individual point views
- **Projection state management**: Each projection maintains its own aggregation level and visualization state

### Performance Optimization
- Lazy loading of GPX files
- Efficient data structures for large datasets
- Client-side caching of filtered results
- Adaptive rendering based on zoom level and selection size

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Usage

```bash
# Start the development server
uv run python -m crossfilter.main serve --port 8000 --preload_jsonl test_data/sample_100.jsonl 

# Or using uvicorn directly
uvicorn crossfilter.main:app serve --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Development

### Code Formatting

```bash
# Format and organize imports for all files (easy way)
./dev/fix_format

# Or run commands manually
uv run --extra dev ruff check --fix . && uv run --extra dev black .

# Check formatting without making changes
uv run --extra dev ruff check . && uv run --extra dev black --check .
```

### Testing
```bash
# Install Playwright browsers (first time only)
uv run --extra dev playwright install
```

```bash
# Run fast tests (skips resource intensive tests like model downloads)
uv run --extra dev pytest -m "not resource_intensive"

# Run all tests including resource intensive ones
uv run --extra dev pytest

# Run with coverage (skipping resource intensive tests)
uv run --extra dev pytest -m "not resource_intensive" --cov=crossfilter

# Run only resource intensive tests (like SIGLIP2 model tests)
uv run --extra dev pytest -m resource_intensive

# Skip frontend tests (useful for CI without browser dependencies)
uv run --extra dev pytest -m "not e2e"


# Disable pytest-xdist for more focused profiling (if needed)
uv run --extra dev py-spy record -o profile.html -- pytest -m "not (resource_intensive or e2e)" -n0
```

### Project Structure
```
crossfilter/
├── core/           # Core data processing and indexing
├── visualization/  # Plotting and rendering components
├── web/           # FastAPI web server and routes
├── static/        # Frontend assets (HTML, CSS, JS)
└── tests/         # Test suite
```

## Data Flow & Projection Architecture

1. **Load**: GPX files are parsed and trackpoints extracted with pre-computed quantized columns
2. **Index**: Points are spatially indexed using H3 and temporally grouped at multiple granularities
3. **Project**: Data is projected into multiple visualization contexts (geographic, temporal, etc.)
4. **Aggregate**: Each projection shows data at appropriate resolution based on max_rows thresholds
5. **Filter**: User selections in any projection update the global filtered dataset
6. **Update**: All projections automatically refresh to reflect the new filtered data
7. **Visualize**: Maps and plots display projection-specific aggregated or individual point data
8. **Cross-filter**: Selections propagate across all projections for coordinated views

## Configuration

The application supports configuration through environment variables:
- `GPX_DATA_DIR`: Directory containing GPX files (default: `./data`)
- `H3_RESOLUTION`: Default H3 resolution level (default: 7)
- `MAX_POINTS_DETAIL`: Threshold for switching to individual point view (default: 100000)
- `SERVER_PORT`: Web server port (default: 8000)

## License

MIT License