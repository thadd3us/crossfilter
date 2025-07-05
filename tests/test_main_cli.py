"""Tests for the main CLI application."""

import socket
import subprocess
import time
from contextlib import closing

import requests
from syrupy.assertion import SnapshotAssertion


def find_free_port() -> int:
    """Find an available port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
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


def test_cli_help_command() -> None:
    """Test that the CLI help command works."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "crossfilter.main", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Crossfilter" in result.stdout
    assert "serve" in result.stdout


def test_serve_command_help() -> None:
    """Test that the serve command help works."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "crossfilter.main", "serve", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout


def test_cli_server_with_preload_jsonl(snapshot: SnapshotAssertion) -> None:
    """Test that the CLI starts the server with --preload-jsonl and data is loaded."""
    port = find_free_port()
    host = "127.0.0.1"
    jsonl_path = "test_data/sample_100.jsonl"

    # Start the server process with preload
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "-m",
            "crossfilter.main",
            "serve",
            "--port",
            str(port),
            "--host",
            host,
            "--preload_jsonl",
            jsonl_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for server to start
        server_started = wait_for_server(host, port, timeout=15.0)
        if not server_started:
            process.terminate()
            process.wait()
            stderr = process.stderr.read() if process.stderr else ""
            stdout = process.stdout.read() if process.stdout else ""
            raise AssertionError(
                f"Server failed to start on {host}:{port}, stderr={stderr}, stdout={stdout}"
            )

        # Give server a moment to fully initialize
        time.sleep(1.0)

        # Make request to the session API to verify data was loaded
        response = requests.get(f"http://{host}:{port}/api/session", timeout=5)
        assert response.status_code == 200

        session_data = response.json()

        # Use snapshot to verify the complete session state
        assert session_data == snapshot

    finally:
        # Clean up: terminate the server process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
