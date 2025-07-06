"""Main CLI application for Crossfilter."""

import json
import logging
import signal
import sys
import traceback
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from crossfilter.core.schema import SchemaColumns as C
import pandas as pd
import typer
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from crossfilter.core.backend_frontend_shared_schema import (
    DfIdsFilterRequest,
    FilterResponse,
    GeoPlotResponse,
    LoadDataRequest,
    LoadDataResponse,
    SessionStateResponse,
    TemporalPlotResponse,
)
from crossfilter.core.backend_frontend_shared_schema import FilterEvent, ProjectionType
from crossfilter.core.schema import load_jsonl_to_dataframe, load_sqlite_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.temporal_cdf_plot import create_temporal_cdf
from crossfilter.visualization.geo_plot import create_geo_plot


@dataclass
class App:
    """Application state container to avoid global variables."""

    session_state: SessionState
    uuid_preview_images_base_dir: Optional[Path] = None


# Create a single app instance for dependency injection
_app_instance = App(session_state=SessionState())

# Configure logging for better error visibility with IDE-clickable file paths
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(pathname)s:%(lineno)d %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_app() -> App:
    """Dependency function to get the app instance."""
    return _app_instance


def get_session_state() -> SessionState:
    """Dependency function to get the session state instance."""
    return _app_instance.session_state


app = FastAPI(
    title="Crossfilter",
    description="Interactive crossfilter application for geospatial and temporal data analysis",
)


# Exception handlers for better error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler that logs full stack traces."""
    # Log the full exception with stack trace
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for HTTP exceptions with logging."""
    logger.error(
        f"HTTP exception in {request.method} {request.url.path}: {exc.status_code} - {exc.detail}",
        exc_info=True,
    )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handler for request validation errors with logging."""
    logger.error(
        f"Validation error in {request.method} {request.url.path}: {exc.errors()}",
        exc_info=True,
    )

    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.post("/api/data/load")
async def load_data_endpoint(
    request: LoadDataRequest, session_state: SessionState = Depends(get_session_state)
) -> LoadDataResponse:
    """Load data from a JSONL file into the session state."""
    try:
        jsonl_path = Path(request.file_path)
        if not jsonl_path.exists():
            raise HTTPException(
                status_code=404, detail=f"File not found: {request.file_path}"
            )

        df = load_jsonl_to_dataframe(jsonl_path)
        session_state.load_dataframe(df)

        return LoadDataResponse(
            success=True,
            message=f"Successfully loaded {len(df)} records",
            session_state=session_state._create_session_state_response(),
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full exception with stack trace
        logger.error(f"Error loading data from {request.file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the main application page."""
    static_file = static_path / "index.html"
    if static_file.exists():
        return static_file.read_text()
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crossfilter - Interactive Data Analysis</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>Crossfilter</h1>
            <p>Interactive crossfilter application for geospatial and temporal data analysis</p>
            <p><strong>Note:</strong> Static files not found. The full interface is not available.</p>
        </body>
        </html>
        """


@app.get("/api/session")
async def get_session_status(
    session_state: SessionState = Depends(get_session_state),
) -> SessionStateResponse:
    """Get the current session state status."""
    return session_state._create_session_state_response()


@app.get("/api/plots/temporal")
async def get_temporal_plot_data(
    max_groups: int = Query(100000, ge=1, le=1000000),
    session_state: SessionState = Depends(get_session_state),
) -> TemporalPlotResponse:
    """Get data for the temporal CDF plot."""
    if len(session_state.all_rows) == 0:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        temporal_data = session_state.get_temporal_projection()
        fig = create_temporal_cdf(
            temporal_data, temporal_projection_state=session_state.temporal_projection
        )
        fig_json = fig.to_json()
        if fig_json is None:
            raise ValueError("Failed to serialize temporal plot to JSON")
        plotly_plot = json.loads(fig_json)

        # Calculate status information
        from crossfilter.core.schema import SchemaColumns as C

        total_row_count = (
            temporal_data[C.COUNT].sum()
            if C.COUNT in temporal_data.columns
            else len(temporal_data)
        )

        return TemporalPlotResponse(
            plotly_plot=plotly_plot,
            bucket_count=len(temporal_data),
            total_row_count=total_row_count,
            aggregation_level=session_state.temporal_projection.current_aggregation_level,
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full exception with stack trace
        logger.error(f"Error generating temporal plot: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating temporal plot: {str(e)}"
        )


@app.get("/api/plots/geo")
async def get_geo_plot_data(
    session_state: SessionState = Depends(get_session_state),
) -> GeoPlotResponse:
    """Get data for the geographic plot."""
    if len(session_state.all_rows) == 0:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        geo_data = session_state.get_geo_aggregation()
        fig = create_geo_plot(
            geo_data, geo_projection_state=session_state.geo_projection
        )
        fig_json = fig.to_json()
        if fig_json is None:
            raise ValueError("Failed to serialize geo plot to JSON")
        plotly_plot = json.loads(fig_json)

        # Calculate status information
        marker_count = len(geo_data)
        from crossfilter.core.schema import SchemaColumns as C

        total_row_count = (
            geo_data[C.COUNT].sum() if C.COUNT in geo_data.columns else len(geo_data)
        )

        # Get aggregation level from geo projection state
        geo_summary = session_state.geo_projection.get_summary()
        h3_level = geo_summary.get("h3_level")
        aggregation_level = f"H3 level {h3_level}" if h3_level is not None else None

        return GeoPlotResponse(
            plotly_plot=plotly_plot,
            bucket_count=len(geo_data),
            total_row_count=total_row_count,
            aggregation_level=aggregation_level,
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full exception with stack trace
        logger.error(f"Error generating geo plot: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating geo plot: {str(e)}"
        )


@app.post("/api/filters/df_ids")
async def filter_to_df_ids(
    request: DfIdsFilterRequest,
    session_state: SessionState = Depends(get_session_state),
) -> FilterResponse:
    """Filter data to only include points with specified df_ids from a plot."""
    logger.info(f"Filtering to df_ids: {request=}")

    try:
        filter_event = FilterEvent(
            projection_type=request.event_source,
            selected_df_ids=set(request.df_ids),
            filter_operator=request.filter_operator,
        )
        session_state.apply_filter_event(filter_event)

        return FilterResponse(
            success=True,
            filter_state=session_state._create_session_state_response(),
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full exception with stack trace
        logger.error(
            f"Error applying {request.event_source.value} filter: {e}", exc_info=True
        )
        # Include the projection type in the error message for better test compatibility
        error_msg = f"Error applying {request.event_source.value} filter: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/filters/reset")
async def reset_filters(
    session_state: SessionState = Depends(get_session_state),
) -> FilterResponse:
    """Reset all filters to show all data."""
    session_state.reset_filters()

    return FilterResponse(
        success=True,
        filter_state=session_state._create_session_state_response(),
    )


@app.get("/api/events/filter-changes")
async def filter_change_stream(
    session_state: SessionState = Depends(get_session_state),
) -> Any:
    """Server-Sent Events stream for filter change notifications."""
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        session_state.filter_change_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@app.get("/api/image_preview/uuid/{uuid}")
async def get_uuid_preview_image(
    uuid: str, app_instance: App = Depends(get_app)
) -> Response:
    """Get a preview image for a UUID."""
    if not app_instance.uuid_preview_images_base_dir:
        return _create_no_preview_available_image()

    # Extract first two characters for subdirectory
    if len(uuid) < 2:
        return _create_no_preview_available_image()

    subdir = uuid[:2]
    image_path = app_instance.uuid_preview_images_base_dir / subdir / f"{uuid}.jpg"

    if not image_path.exists():
        return _create_no_preview_available_image()

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        return Response(content=image_data, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error reading image file {image_path}: {e}")
        return _create_no_preview_available_image()


def _create_no_preview_available_image() -> Response:
    """Create a dummy image with 'No preview available' text."""
    # Create a simple SVG image
    svg_content = """<svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="150" fill="#f0f0f0" stroke="#ccc" stroke-width="2"/>
        <text x="100" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#666">
            No preview available
        </text>
    </svg>"""

    return Response(content=svg_content, media_type="image/svg+xml")


# https://github.com/fastapi/typer/issues/341
typer.main.get_command_name = lambda name: name

cli = typer.Typer(
    help="Crossfilter - Interactive crossfilter application for geospatial and temporal data analysis",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


@cli.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Crossfilter - Interactive crossfilter application for geospatial and temporal data analysis."""
    pass


@cli.command("serve")
def serve(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    preload_jsonl: Optional[Path] = typer.Option(
        None,
        help="Path to JSONL file to preload into session state",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    preload_sqlite_db: Optional[Path] = typer.Option(
        None,
        help="Path to SQLite database file to preload into session state",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    preload_sqlite_table: str = typer.Option(
        "data", help="Table name in SQLite database to preload (default: 'data')"
    ),
    uuid_preview_images_base_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing UUID preview images organized in subdirectories",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
) -> None:
    """Start the Crossfilter web application."""

    # Set UUID preview images base directory on app instance
    _app_instance.uuid_preview_images_base_dir = uuid_preview_images_base_dir

    # Handle preload data if provided
    dataframes = []

    if preload_jsonl:
        if not preload_jsonl.exists():
            typer.echo(f"Error: JSONL file '{preload_jsonl}' does not exist.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading data from {preload_jsonl}...")
        df_jsonl = load_jsonl_to_dataframe(preload_jsonl)
        dataframes.append(df_jsonl)
        typer.echo(f"Successfully loaded {len(df_jsonl)} records from JSONL")

    if preload_sqlite_db:
        typer.echo(
            f"Loading data from {preload_sqlite_db} table '{preload_sqlite_table}'..."
        )
        df_sqlite = load_sqlite_to_dataframe(preload_sqlite_db, preload_sqlite_table)
        dataframes.append(df_sqlite)
        typer.echo(f"Successfully loaded {len(df_sqlite)} records from SQLite")

    # Allow starting without data for UI testing
    if dataframes:
        final_df = pd.concat(dataframes, axis="index", ignore_index=True)
        logger.info(
            f"Concatenated {len(dataframes)=} dataframes into {final_df.shape=}"
        )
        final_df = final_df.reset_index(drop=True)
        final_df.index.name = C.DF_ID
        _app_instance.session_state.load_dataframe(final_df)
    else:
        logger.info("Starting server without preloaded data")

    def signal_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        typer.echo(
            f"Shutting down Crossfilter {signal.Signals(signum).name=}, {frame=}..."
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    typer.echo(f"Starting Crossfilter on http://{host}:{port}")

    # Use Uvicorn as the ASGI server - it's the recommended production server for FastAPI
    # providing high performance async request handling
    # Pass the app instance directly to preserve the session state
    # Configure for faster shutdown during testing
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        # Reduce shutdown timeout for faster test execution
        # This affects how long uvicorn waits for connections to close
        timeout_graceful_shutdown=2,
        # Reduce keep-alive timeout to prevent hanging connections
        timeout_keep_alive=1,
    )


def main() -> None:
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()
