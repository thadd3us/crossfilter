"""Main CLI application for GPX Viewer."""

import asyncio
import signal
import sys
from typing import Optional

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI(title="GPX Viewer", description="Interactive GPX file viewer with cross-filtering capabilities")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the main application page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPX Viewer - Crossfilter Interactive Maps</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <h1>GPX Viewer</h1>
        <p>Interactive GPX file viewer with Crossfilter capabilities</p>
    </body>
    </html>
    """


cli = typer.Typer(help="GPX Viewer - Interactive GPX file viewer with cross-filtering")


@cli.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the GPX Viewer web application."""
    
    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        typer.echo("Shutting down GPX Viewer...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    typer.echo(f"Starting GPX Viewer on http://{host}:{port}")
    
    uvicorn.run(
        "gpx_viewer.main:app",
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