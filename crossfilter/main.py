"""Main CLI application for Crossfilter."""

import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import typer
import uvicorn
from fastapi import Depends, FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
from crossfilter.core.schema import load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.temporal_cdf_plot import create_temporal_cdf
from crossfilter.visualization.geo_plot import create_geo_plot

# Create a single session state instance for dependency injection
_session_state_instance = SessionState()

# Configure logging for better error visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_session_state() -> SessionState:
    """Dependency function to get the session state instance."""
    return _session_state_instance


app = FastAPI(
    title="Crossfilter",
    description="Interactive crossfilter application for geospatial and temporal data analysis",
)





# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")




@app.post("/api/data/load")
async def load_data_endpoint(
    request: LoadDataRequest, session_state: SessionState = Depends(get_session_state)
) -> LoadDataResponse:
    """Load data from a JSONL file into the session state."""
    from fastapi import HTTPException

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
    except Exception as e:
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
    from fastapi import HTTPException

    if len(session_state.all_rows) == 0:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        temporal_data = session_state.get_temporal_projection()
        fig = create_temporal_cdf(temporal_data)
        plotly_plot = json.loads(fig.to_json())

        # Calculate status information
        from crossfilter.core.schema import SchemaColumns as C
        distinct_point_count = temporal_data[C.COUNT].sum() if C.COUNT in temporal_data.columns else len(temporal_data)
        
        # Get aggregation level from temporal projection state
        temporal_summary = session_state.temporal_projection.get_summary()
        target_column = temporal_summary.get("target_column")
        if target_column and "temporal_" in target_column:
            # Extract granularity from column name like "temporal_minute", "temporal_hour", etc.
            granularity = target_column.replace("temporal_", "").capitalize()
            aggregation_level = f"{granularity} buckets"
        else:
            aggregation_level = None

        return TemporalPlotResponse(
            plotly_plot=plotly_plot,
            data_type="aggregated" if C.COUNT in temporal_data.columns else "individual",
            point_count=len(temporal_data),
            distinct_point_count=distinct_point_count,
            aggregation_level=aggregation_level,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating temporal plot: {str(e)}"
        )


@app.get("/api/plots/geo")
async def get_geo_plot_data(
    session_state: SessionState = Depends(get_session_state),
) -> GeoPlotResponse:
    """Get data for the geographic plot."""
    from fastapi import HTTPException

    if len(session_state.all_rows) == 0:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        geo_data = session_state.get_geo_aggregation()
        fig = create_geo_plot(geo_data)
        plotly_plot = json.loads(fig.to_json())

        # Calculate status information
        marker_count = len(geo_data)
        from crossfilter.core.schema import SchemaColumns as C
        distinct_point_count = geo_data[C.COUNT].sum() if C.COUNT in geo_data.columns else len(geo_data)
        
        # Get aggregation level from geo projection state
        geo_summary = session_state.geo_projection.get_summary()
        h3_level = geo_summary.get("h3_level")
        aggregation_level = f"H3 level {h3_level}" if h3_level is not None else None

        return GeoPlotResponse(
            plotly_plot=plotly_plot,
            marker_count=marker_count,
            distinct_point_count=distinct_point_count,
            aggregation_level=aggregation_level,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating geo plot: {str(e)}"
        )


@app.post("/api/filters/df_ids")
async def filter_to_df_ids(
    request: DfIdsFilterRequest,
    session_state: SessionState = Depends(get_session_state),
) -> FilterResponse:
    """Filter data to only include points with specified df_ids from a plot."""
    from fastapi import HTTPException

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
    except Exception as e:
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
):
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








# https://github.com/fastapi/typer/issues/341
typer.main.get_command_name = lambda name: name

cli = typer.Typer(
    help="Crossfilter - Interactive crossfilter application for geospatial and temporal data analysis"
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
        None, help="Path to JSONL file to preload into session state"
    ),
) -> None:
    """Start the Crossfilter web application."""

    # Handle preload data if provided
    if preload_jsonl:
        if not preload_jsonl.exists():
            typer.echo(f"Error: JSONL file '{preload_jsonl}' does not exist.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading data from {preload_jsonl}...")
        try:
            df = load_jsonl_to_dataframe(preload_jsonl)
            _session_state_instance.load_dataframe(df)
            typer.echo(f"Successfully loaded {len(df)} records")
        except Exception as e:
            typer.echo(f"Error loading data: {str(e)}", err=True)
            raise typer.Exit(1)

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
