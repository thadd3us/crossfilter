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
    
    # Navigate to the main application page
    page.goto(f"{server_url}/")
    
    # Wait for the page title to be set
    page.wait_for_function("document.title === 'Crossfilter - Temporal CDF Analysis'")
    
    # Wait for the app to initialize and detect pre-loaded data
    page.wait_for_function("document.getElementById('refreshBtn').disabled === false", timeout=10000)
    
    # Click the refresh plot button to load the temporal CDF plot
    page.click("#refreshBtn")
    
    # Wait for the plot to be rendered in the plot container
    page.wait_for_function("""
        () => {
            const plotContainer = document.getElementById('plotContainer');
            return plotContainer && plotContainer.querySelector('.main-svg') !== null;
        }
    """, timeout=30000)
    
    # Wait a bit more for the plot to fully render
    page.wait_for_timeout(2000)
    
    # Take a screenshot of the entire page
    screenshot_bytes = page.screenshot(full_page=True)
    
    # Use syrupy to compare the screenshot
    assert screenshot_bytes == snapshot(extension_class=TemporalCDFPNGExtension)


@pytest.mark.e2e
def test_temporal_cdf_plot_content(
    page: Page, 
    backend_server_with_data: str
) -> None:
    """Test that the main application page contains expected content and loads the plot."""
    server_url = backend_server_with_data
    
    # Navigate to the main application page
    page.goto(f"{server_url}/")
    
    # Check the page title
    assert page.title() == "Crossfilter - Temporal CDF Analysis"
    
    # Check the main heading
    heading = page.locator("h1").text_content()
    assert heading == "Crossfilter - Temporal CDF Analysis"
    
    # Check the subtitle
    subtitle = page.locator("p").text_content()
    assert subtitle == "Interactive temporal analysis with cumulative distribution functions"
    
    # Wait for the app to initialize and detect pre-loaded data
    page.wait_for_function("document.getElementById('refreshBtn').disabled === false", timeout=10000)
    
    # Verify that status shows data is loaded
    status_text = page.locator("#status").text_content()
    assert "Data loaded" in status_text
    assert "100 rows" in status_text
    
    # Click the refresh plot button to load the temporal CDF plot
    page.click("#refreshBtn")
    
    # Wait for the plot to be rendered in the plot container
    page.wait_for_function("""
        () => {
            const plotContainer = document.getElementById('plotContainer');
            return plotContainer && plotContainer.querySelector('.main-svg') !== null;
        }
    """, timeout=30000)
    
    # Verify the plot container exists and has content
    plot_container = page.locator("#plotContainer")
    assert plot_container.count() > 0
    assert plot_container.locator(".main-svg").count() > 0