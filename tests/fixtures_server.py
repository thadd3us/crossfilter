"""Common server fixtures for testing."""

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import requests

logger = logging.getLogger(__name__)


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
def server_with_data() -> Generator[str, None, None]:
    """Start the backend server with pre-loaded sample data.
    
    This fixture provides comprehensive monitoring with real-time backend log output,
    making it easier to debug issues during testing.
    """
    # Path to the sample data
    sample_data_path = (
        Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    )

    # Start the server on a test port
    test_port = 8001
    server_url = f"http://localhost:{test_port}"

    # Command to start the server with pre-loaded data
    cmd = (
        sys.executable,
        *("-m", "crossfilter.main", "serve"),
        *("--port", str(test_port)),
        *("--preload_jsonl", str(sample_data_path)),
    )

    # Start the server process
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier monitoring
        cwd=Path(__file__).parent.parent,
        text=True,
        bufsize=1,  # Line buffered for real-time output
    )

    # Set up monitoring for server output
    server_output_lines = []

    def monitor_server_output():
        """Monitor server output in background thread."""
        try:
            for line in iter(server_process.stdout.readline, ""):
                if line:
                    server_output_lines.append(line.strip())
                    # Log server output in real-time for debugging
                    logger.info(f"[BACKEND] {line.strip()}")
        except Exception as e:
            logger.error(f"[BACKEND MONITOR ERROR] {e}")

    monitor_thread = threading.Thread(target=monitor_server_output, daemon=True)
    monitor_thread.start()

    try:
        # Wait for server to be ready
        if not wait_for_server(server_url):
            # Give the monitor thread a moment to capture output
            time.sleep(2)
            server_process.terminate()
            monitor_thread.join(timeout=3)

            pytest.fail(
                f"Server failed to start within timeout at {server_url}.\n"
                "Server output:\n"
                + "\n".join(server_output_lines[-50:])  # Show last 50 lines
            )

        yield server_url

    finally:
        # Clean up: terminate the server immediately for faster tests
        logger.info("[BACKEND] Shutting down server...")
        server_process.kill()
        server_process.wait()

        # Wait for monitor thread to finish
        monitor_thread.join(timeout=3)
