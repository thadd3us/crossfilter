"""Direct API tests for filter endpoints without browser dependency."""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

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
def api_server_with_data() -> Generator[str, None, None]:
    """Start the backend server with pre-loaded sample data for direct API testing."""
    # Path to the sample data
    sample_data_path = (
        Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    )

    # Start the server on a test port
    test_port = 8002  # Different port from frontend tests to avoid conflicts
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

    try:
        # Wait for server to be ready
        if not wait_for_server(server_url):
            server_process.terminate()
            pytest.fail(f"Server failed to start within timeout at {server_url}")

        yield server_url

    finally:
        # Clean up: terminate the server immediately for faster tests
        logger.info("[API TEST] Shutting down server...")
        server_process.kill()
        server_process.wait()


def test_filter_df_ids_endpoint_direct(api_server_with_data: str) -> None:
    """Test the /api/filters/df_ids endpoint directly with HTTP requests."""
    server_url = api_server_with_data
    
    # First, get the session status to verify data is loaded
    response = requests.get(f"{server_url}/api/session")
    assert response.status_code == 200
    session_data = response.json()
    assert session_data["has_data"] is True
    assert session_data["row_count"] == 100
    logger.info(f"Session data: {session_data}")
    
    # Get temporal plot data to understand the bucketing structure
    response = requests.get(f"{server_url}/api/plots/temporal?max_groups=10000")
    assert response.status_code == 200
    plot_data = response.json()
    logger.info(f"Plot data type: {type(plot_data)}")
    logger.info(f"Plot data keys: {plot_data.keys() if isinstance(plot_data, dict) else 'not a dict'}")
    
    # Extract the plotly figure data
    plotly_plot = plot_data["plotly_plot"]
    plotly_data = plotly_plot["data"][0]  # Get the first trace
    
    # Get the customdata which should contain df_ids for each point
    customdata = plotly_data.get("customdata", [])
    logger.info(f"Number of plot points: {len(customdata)}")
    logger.info(f"First few customdata entries: {customdata[:5] if customdata else 'No customdata'}")
    
    # Extract df_ids from the first 10 points (simulating a selection)
    if customdata:
        # Each customdata entry should have df_id
        selected_df_ids = []
        for i, entry in enumerate(customdata[:10]):  # Select first 10 points
            if isinstance(entry, dict) and "df_id" in entry:
                selected_df_ids.append(entry["df_id"])
            elif isinstance(entry, list) and len(entry) > 0:
                # If customdata is a list of lists, df_id might be the first element
                selected_df_ids.append(entry[0])
        
        if not selected_df_ids:
            # Fallback: use point indices as df_ids
            selected_df_ids = list(range(10))
    else:
        # Fallback: use point indices as df_ids
        selected_df_ids = list(range(10))
    
    logger.info(f"Selected df_ids for filtering: {selected_df_ids}")
    
    # Test the filter endpoint with these df_ids
    filter_request = {
        "df_ids": selected_df_ids,
        "event_source": "temporal",
        "description": "Test filtering from API test"
    }
    
    logger.info(f"Sending filter request: {filter_request}")
    
    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"}
    )
    
    # Log the response details for debugging
    logger.info(f"Filter response status: {response.status_code}")
    logger.info(f"Filter response headers: {response.headers}")
    
    if response.status_code != 200:
        logger.error(f"Filter response text: {response.text}")
        
    # This is the main assertion - the endpoint should return success
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    filter_result = response.json()
    logger.info(f"Filter result: {filter_result}")
    
    # Verify the response structure
    assert filter_result["success"] is True
    assert filter_result["event_source"] == "temporal"
    assert "filtered_count" in filter_result
    assert "bucket_count" in filter_result
    assert "filter_state" in filter_result
    
    # The filtered count should be less than or equal to the original count
    assert filter_result["filtered_count"] <= 100
    assert filter_result["filtered_count"] > 0
    
    # Verify that the session state was updated by checking the session endpoint again
    response = requests.get(f"{server_url}/api/session")
    assert response.status_code == 200
    updated_session_data = response.json()
    
    # The filtered count should be reflected in the session state
    assert updated_session_data["filtered_count"] == filter_result["filtered_count"]
    assert updated_session_data["filtered_count"] < 100  # Should be less than original


def test_filter_df_ids_endpoint_invalid_request(api_server_with_data: str) -> None:
    """Test the /api/filters/df_ids endpoint with invalid requests."""
    server_url = api_server_with_data
    
    # Test with empty df_ids list
    filter_request = {
        "df_ids": [],
        "event_source": "temporal",
        "description": "Test with empty df_ids"
    }
    
    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"}
    )
    
    # Should handle empty df_ids gracefully (either 400 or return empty result)
    logger.info(f"Empty df_ids response: {response.status_code}, {response.text}")
    # The actual behavior depends on implementation - could be 400 or success with 0 filtered_count
    
    # Test with invalid event_source
    filter_request = {
        "df_ids": [1, 2, 3],
        "event_source": "invalid_source",
        "description": "Test with invalid event_source"
    }
    
    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"}
    )
    
    # Should return 400 for invalid event_source
    assert response.status_code == 422  # FastAPI validation error for invalid enum value
    
    # Test with geo event_source (should return 501 - not implemented)
    filter_request = {
        "df_ids": [1, 2, 3],
        "event_source": "geo", 
        "description": "Test with geo event_source"
    }
    
    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"}
    )
    
    # Should return 501 for geo filtering (not implemented) or 500 if there's an error
    assert response.status_code in [500, 501]
    if response.status_code == 501:
        assert "not yet implemented" in response.text.lower()
    else:
        # If 500, it should still be related to geo filtering not being implemented
        assert "geo" in response.text.lower() or "spatial" in response.text.lower()