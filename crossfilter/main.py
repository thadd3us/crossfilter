"""Main CLI application for Crossfilter."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from crossfilter.core.session_state import SessionState
from crossfilter.core.data_schema import load_jsonl_to_dataframe
from crossfilter.visualization.plots import PlotGenerator


app = FastAPI(title="Crossfilter", description="Interactive crossfilter application for geospatial and temporal data analysis")

# Global session state - single instance for the entire application
session_state = SessionState()

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# THAD: Never create a global variable for anything.
# Global variable to hold preload path for startup event
_preload_jsonl_path: Optional[Path] = None


@app.post("/api/data/load")
async def load_data_endpoint(file_path: str) -> Dict[str, Any]:
    """Load data from a JSONL file into the session state."""
    try:
        jsonl_path = Path(file_path)
        if not jsonl_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        df = load_jsonl_to_dataframe(jsonl_path)
        session_state.load_dataframe(df)
        
        return {
            "success": True,
            "message": f"Successfully loaded {len(df)} records",
            "session_state": session_state.get_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


def set_preload_path(path: Path) -> None:
    """Set the path for data to be loaded during startup."""
    global _preload_jsonl_path
    _preload_jsonl_path = path


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
async def get_session_status() -> dict:
    """Get the current session state status."""
    summary = session_state.get_summary()
    print(f"DEBUG: Session status requested - has_data: {session_state.has_data()}")
    return summary


# Pydantic models for API requests
class FilterRequest(BaseModel):
    """Request model for applying filters."""
    uuids: List[int]
    operation_type: str  # 'spatial' or 'temporal'
    description: str
    metadata: Optional[Dict[str, Any]] = None


@app.get("/api/plots/spatial")
async def get_spatial_plot_data(max_groups: int = Query(100000, ge=1, le=1000000)) -> Dict[str, Any]:
    """Get data for the spatial heatmap plot."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        # Get aggregated spatial data
        spatial_data = session_state.get_spatial_aggregation(max_groups)
        
        # Prepare data and config for Kepler.gl
        kepler_data = PlotGenerator.prepare_kepler_data(spatial_data)
        kepler_config = PlotGenerator.create_kepler_config(spatial_data)
        
        # Also create fallback Plotly plot
        plotly_plot = PlotGenerator.create_fallback_scatter_geo(spatial_data)
        
        return {
            "kepler_data": kepler_data,
            "kepler_config": kepler_config,
            "plotly_fallback": plotly_plot,
            "data_type": "aggregated" if 'count' in spatial_data.columns else "individual",
            "point_count": len(spatial_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating spatial plot: {str(e)}")


@app.get("/api/plots/temporal")
async def get_temporal_plot_data(max_groups: int = Query(100000, ge=1, le=1000000)) -> Dict[str, Any]:
    """Get data for the temporal CDF plot."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        # Get aggregated temporal data
        temporal_data = session_state.get_temporal_aggregation(max_groups)
        
        # Create Plotly CDF plot
        plotly_plot = PlotGenerator.create_temporal_cdf(temporal_data)
        
        return {
            "plotly_plot": plotly_plot,
            "data_type": "aggregated" if 'count' in temporal_data.columns else "individual", 
            "point_count": len(temporal_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating temporal plot: {str(e)}")


@app.post("/api/filters/apply")
async def apply_filter(filter_request: FilterRequest) -> Dict[str, Any]:
    """Apply a spatial or temporal filter."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        # Apply filter based on type
        # THAD: There should be very few naked strings like this.  These should be enum values.
        if filter_request.operation_type == 'spatial':
            session_state.filter_state.apply_spatial_filter(
                set(filter_request.uuids),
                filter_request.description,
                filter_request.metadata
            )
        elif filter_request.operation_type == 'temporal':
            session_state.filter_state.apply_temporal_filter(
                set(filter_request.uuids),
                filter_request.description,
                filter_request.metadata
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid operation_type. Must be 'spatial' or 'temporal'")
        
        return {
            "success": True,
            "filter_state": session_state.filter_state.get_summary()
        }
    except Exception as e:
        # Thad: I'm seeing a linter suggestion here about re-raising
        raise HTTPException(status_code=500, detail=f"Error applying filter: {str(e)}")


@app.post("/api/filters/intersect")
async def intersect_filter(filter_request: FilterRequest) -> Dict[str, Any]:
    """Intersect current filter with new selection."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        session_state.filter_state.intersect_with_filter(
            set(filter_request.uuids),
            filter_request.operation_type,
            filter_request.description,
            filter_request.metadata
        )
        
        return {
            "success": True,
            "filter_state": session_state.filter_state.get_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error intersecting filter: {str(e)}")


@app.post("/api/filters/reset")
async def reset_filters() -> Dict[str, Any]:
    """Reset all filters to show all data."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        session_state.filter_state.reset_filters()
        
        return {
            "success": True,
            "filter_state": session_state.filter_state.get_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting filters: {str(e)}")


@app.post("/api/filters/undo")
async def undo_filter() -> Dict[str, Any]:
    """Undo the last filter operation."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        success = session_state.filter_state.undo()
        
        return {
            "success": success,
            "filter_state": session_state.filter_state.get_summary(),
            "message": "Filter undone" if success else "No operations to undo"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error undoing filter: {str(e)}")


@app.get("/api/filters/history")
async def get_filter_history() -> Dict[str, Any]:
    """Get the filter operation history."""
    if not session_state.has_data():
        raise HTTPException(status_code=404, detail="No data loaded")
    
    try:
        history = session_state.filter_state.get_undo_stack_info()
        
        return {
            "history": history,
            "can_undo": session_state.filter_state.can_undo
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting filter history: {str(e)}")


cli = typer.Typer(help="Crossfilter - Interactive crossfilter application for geospatial and temporal data analysis")


@cli.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Crossfilter - Interactive crossfilter application for geospatial and temporal data analysis."""
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, show help
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@cli.command("serve")
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    preload_jsonl: Optional[Path] = typer.Option(None, "--preload-jsonl", help="Path to JSONL file to preload into session state"),
) -> None:
    """Start the Crossfilter web application."""
    
    # Set preload data path if JSONL file is provided
    if preload_jsonl:
        if not preload_jsonl.exists():
            typer.echo(f"Error: JSONL file '{preload_jsonl}' does not exist.", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Will load data from {preload_jsonl} during server startup...")
        set_preload_path(preload_jsonl)
    
    # THAD: Add type annotations to all function arguments.
    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        typer.echo("Shutting down Crossfilter...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    typer.echo(f"Starting Crossfilter on http://{host}:{port}")
    
    # THAD: Explain what uvicorn is and why we're using it.
    # THAD: Is there a way to pass the preload_jsonl object on here?
    uvicorn.run(
        "crossfilter.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main() -> None:
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()