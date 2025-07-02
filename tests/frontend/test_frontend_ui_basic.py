"""Basic frontend UI test for temporal CDF plot using Playwright."""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Generator

import pytest
import requests
from playwright.sync_api import Page
from syrupy import SnapshotAssertion
from syrupy.extensions.image import PNGImageSnapshotExtension


class TemporalCDFPNGExtension(PNGImageSnapshotExtension):
    """Custom PNG snapshot extension for temporal CDF plots."""

    _file_extension = "png"


def wait_for_server(url: str, max_attempts: int = 30, delay: float = 1.0) -> bool:
    """Wait for the server to be ready."""
    for _ in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    return False


@pytest.fixture(scope="function")
def backend_server_with_data() -> Generator[str, None, None]:
    """Start the backend server with pre-loaded sample data."""
    # Path to the sample data
    sample_data_path = Path(__file__).parent.parent.parent / "test_data" / "sample_100.jsonl"
    
    if not sample_data_path.exists():
        pytest.skip(f"Sample data file not found: {sample_data_path}")
    
    # Start the server on a test port
    test_port = 8001
    server_url = f"http://localhost:{test_port}"
    
    # Command to start the server with pre-loaded data
    cmd = [
        sys.executable, "-m", "crossfilter.main", "serve",
        "--port", str(test_port),
        "--host", "127.0.0.1",
        "--preload_jsonl", str(sample_data_path)
    ]
    
    # Start the server process
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    try:
        # Wait for server to be ready
        if not wait_for_server(server_url):
            stdout, stderr = server_process.communicate(timeout=5)
            pytest.fail(
                f"Server failed to start within timeout.\n"
                f"STDOUT: {stdout.decode()}\n"
                f"STDERR: {stderr.decode()}"
            )
        
        # Verify data is loaded
        session_response = requests.get(f"{server_url}/api/session")
        if not session_response.ok:
            pytest.fail(f"Session endpoint not accessible: {session_response.status_code}")
        
        session_data = session_response.json()
        if not session_data.get("has_data"):
            pytest.fail("Server started but no data was loaded")
        
        print(f"Server started successfully with {session_data.get('row_count', 0)} records")
        yield server_url
        
    finally:
        # Clean up: terminate the server
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()


@pytest.mark.e2e
def test_temporal_cdf_plot_display(
    page: Page, 
    backend_server_with_data: str,
    snapshot: SnapshotAssertion
) -> None:
    """Test that the temporal CDF plot loads and displays correctly."""
    server_url = backend_server_with_data
    
    # Navigate to the temporal CDF page
    page.goto(f"{server_url}/static/temporal-cdf.html")
    
    # Wait for the page title to be set
    page.wait_for_function("document.title === 'Temporal CDF Plot'")
    
    # Wait for the plot div to be created
    page.wait_for_selector("#temporal-cdf-plot", state="attached", timeout=30000)
    
    # Wait for the fetch to complete and plot to be loaded
    page.wait_for_timeout(3000)
    
    # Take a screenshot of the entire page
    screenshot_bytes = page.screenshot(full_page=True)
    
    # Use syrupy to compare the screenshot
    assert screenshot_bytes == snapshot(extension_class=TemporalCDFPNGExtension)


@pytest.mark.e2e
def test_temporal_cdf_plot_content(
    page: Page, 
    backend_server_with_data: str
) -> None:
    """Test that the temporal CDF plot page contains expected content."""
    server_url = backend_server_with_data
    
    # Navigate to the temporal CDF page
    page.goto(f"{server_url}/static/temporal-cdf.html")
    
    # Check the page title
    assert page.title() == "Temporal CDF Plot"
    
    # Check the main heading
    heading = page.locator("h1").text_content()
    assert heading == "Temporal CDF Plot"
    
    # Check the subtitle
    subtitle = page.locator("p").text_content()
    assert subtitle == "Cumulative distribution function of temporal data"
    
    # Wait for the plot div to be created
    page.wait_for_selector("#temporal-cdf-plot", state="attached", timeout=30000)
    
    # Wait for the fetch to complete and plot to be loaded
    page.wait_for_timeout(3000)
    
    # Verify the plot container exists
    plot_container = page.locator("#temporal-cdf-plot")
    assert plot_container.count() > 0