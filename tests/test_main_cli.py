"""Tests for the main CLI application."""

import socket
import subprocess
import time
from contextlib import closing

import pytest
import requests


def find_free_port() -> int:
    """Find an available port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for server to start accepting connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                if result == 0:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def test_cli_server_startup_and_html_content() -> None:
    """Test that the CLI starts the server and serves HTML with 'Crossfilter' in title."""
    port = find_free_port()
    host = "127.0.0.1"
    
    # Start the server process
    process = subprocess.Popen([
        "uv", "run", "python", "-m", "gpx_viewer.main", "serve", 
        "--port", str(port), "--host", host
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server to start
        server_started = wait_for_server(host, port, timeout=10.0)
        if not server_started:
            process.terminate()
            process.wait()
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ''
            raise AssertionError(f"Server failed to start on {host}:{port}, {stderr=}")
        
        # Give server a moment to fully initialize
        time.sleep(0.5)
        
        # Make request to the main page
        response = requests.get(f"http://{host}:{port}/", timeout=5)
        assert response.status_code == 200
        
        # Check that the HTML contains "Crossfilter" in the title
        html_content = response.text
        assert "Crossfilter" in html_content, f"'Crossfilter' not found in HTML content: {html_content}"
        
        # Additional checks for expected content
        assert "<title>" in html_content
        assert "Crossfilter" in html_content
        
    finally:
        # Clean up: terminate the server process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_cli_help_command():
    """Test that the CLI help command works."""
    result = subprocess.run([
        "uv", "run", "python", "-m", "gpx_viewer.main", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Crossfilter" in result.stdout
    assert "serve" in result.stdout


def test_serve_command_help():
    """Test that the serve command help works."""
    result = subprocess.run([
        "uv", "run", "python", "-m", "gpx_viewer.main", "serve", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout