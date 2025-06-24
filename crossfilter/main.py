"""Main CLI application for Crossfilter."""

import asyncio
import signal
import sys
from typing import Optional

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from crossfilter.core.session_state import SessionState


app = FastAPI(title="Crossfilter", description="Interactive crossfilter application for geospatial and temporal data analysis")

# Global session state - single instance for the entire application
session_state = SessionState()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the main application page."""
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
    </body>
    </html>
    """


@app.get("/api/session")
async def get_session_status() -> dict:
    """Get the current session state status."""
    return session_state.get_summary()


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
) -> None:
    """Start the Crossfilter web application."""
    
    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        typer.echo("Shutting down Crossfilter...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    typer.echo(f"Starting Crossfilter on http://{host}:{port}")
    
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