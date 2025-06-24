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

### Data Processing
- **GPX parsing**: Extracts trackpoints with coordinates, timestamps, and metadata
- **Spatial indexing**: H3 hexagonal indexing for efficient spatial queries and aggregation
- **Temporal grouping**: Minute/hour/day/month aggregation for temporal analysis
- **Adaptive resolution**: Automatically adjusts visualization detail based on data size

### Visualization
- **Heatmap**: Geospatial density visualization using Plotly/Kepler.gl
- **CDF plot**: Cumulative distribution function of timestamps
- **Cross-filtering**: Bidirectional selection between spatial and temporal views
- **Progressive detail**: Switch between aggregated and individual point views

### Performance Optimization
- Lazy loading of GPX files
- Efficient data structures for large datasets
- Client-side caching of filtered results
- Adaptive rendering based on zoom level and selection size

## Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Usage

```bash
# Start the development server
uv run python -m crossfilter.main

# Or using uvicorn directly
uvicorn crossfilter.main:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Development

### Testing
```bash
# Run all tests
uv run --extra dev pytest

# Run with coverage
uv run --extra dev pytest --cov=crossfilter
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

## Data Flow

1. **Load**: GPX files are parsed and trackpoints extracted
2. **Index**: Points are spatially indexed using H3 and temporally grouped
3. **Aggregate**: Initial view shows aggregated data at appropriate resolution
4. **Filter**: User selections refine the active point set
5. **Visualize**: Maps and plots update to show filtered results
6. **Annotate**: Individual points can be selected and annotated when visible

## Configuration

The application supports configuration through environment variables:
- `GPX_DATA_DIR`: Directory containing GPX files (default: `./data`)
- `H3_RESOLUTION`: Default H3 resolution level (default: 7)
- `MAX_POINTS_DETAIL`: Threshold for switching to individual point view (default: 100000)
- `SERVER_PORT`: Web server port (default: 8000)

## License

MIT License