"""Direct API tests for filter endpoints without browser dependency."""

import logging

import requests
from syrupy import SnapshotAssertion

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)


def test_filter_df_ids_endpoint_direct(
    server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test the /api/filters/df_ids endpoint directly with HTTP requests."""
    server_url = server_with_data

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
    logger.info(
        f"Plot data keys: {plot_data.keys() if isinstance(plot_data, dict) else 'not a dict'}"
    )

    # Extract the plotly figure data
    plotly_plot = plot_data["plotly_plot"]
    plotly_data = plotly_plot["data"][0]  # Get the first trace

    # Get the customdata which should contain df_ids for each point
    customdata = plotly_data.get("customdata", [])
    logger.info(f"Number of plot points: {len(customdata)}")
    logger.info(
        f"First few customdata entries: {customdata[:5] if customdata else 'No customdata'}"
    )

    # Extract df_ids from the first 10 points (simulating a selection)
    if customdata:
        # Each customdata entry should have df_id
        selected_df_ids = []
        for _i, entry in enumerate(customdata[:10]):  # Select first 10 points
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
        "filter_operator": "intersection",
    }

    logger.info(f"Sending filter request: {filter_request}")

    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"},
    )

    # Log the response details for debugging
    logger.info(f"Filter response status: {response.status_code}")
    logger.info(f"Filter response headers: {response.headers}")

    if response.status_code != 200:
        logger.error(f"Filter response text: {response.text}")

    # This is the main assertion - the endpoint should return success
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    filter_result = response.json()
    logger.info(f"Filter result: {filter_result}")

    assert response.status_code == 200
    assert filter_result == snapshot

    # Test subtraction operation as well
    logger.info("Testing subtraction filter operation...")
    
    # Reset filters first by calling the reset endpoint
    reset_response = requests.post(f"{server_url}/api/filters/reset")
    assert reset_response.status_code == 200
    
    # Test subtraction filter with some df_ids  
    subtraction_request = {
        "df_ids": selected_df_ids[:5],  # Remove first 5 selected points
        "event_source": "temporal",
        "filter_operator": "subtraction",
    }
    
    subtraction_response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=subtraction_request,
        headers={"Content-Type": "application/json"},
    )
    
    assert subtraction_response.status_code == 200
    subtraction_result = subtraction_response.json()
    assert subtraction_result["success"] is True
    
    # After subtraction, we should have 100 - 5 = 95 rows remaining
    assert subtraction_result["filter_state"]["filtered_count"] == 95

    # # Verify the response structure
    # assert filter_result["success"] is True
    # assert filter_result["event_source"] == "temporal"
    # assert "filtered_count" in filter_result
    # assert "bucket_count" in filter_result
    # assert "filter_state" in filter_result

    # # The filtered count should be less than or equal to the original count
    # assert filter_result["filtered_count"] <= 100
    # assert filter_result["filtered_count"] > 0

    # # Verify that the session state was updated by checking the session endpoint again
    # response = requests.get(f"{server_url}/api/session")
    # assert response.status_code == 200
    # updated_session_data = response.json()

    # # The filtered count should be reflected in the session state
    # assert updated_session_data["filtered_count"] == filter_result["filtered_count"]
    # assert updated_session_data["filtered_count"] < 100  # Should be less than original


def test_filter_df_ids_endpoint_invalid_request(server_with_data: str) -> None:
    """Test the /api/filters/df_ids endpoint with invalid requests."""
    server_url = server_with_data

    # Test with empty df_ids list
    filter_request = {
        "df_ids": [],
        "event_source": "temporal",
        "filter_operator": "intersection",
    }

    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"},
    )

    # Should handle empty df_ids gracefully (either 400 or return empty result)
    logger.info(f"Empty df_ids response: {response.status_code}, {response.text}")
    # The actual behavior depends on implementation - could be 400 or success with 0 filtered_count

    # Test with invalid event_source
    filter_request = {
        "df_ids": [1, 2, 3],
        "event_source": "invalid_source",
        "filter_operator": "intersection",
    }

    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"},
    )

    # Should return 400 for invalid event_source
    assert (
        response.status_code == 422
    )  # FastAPI validation error for invalid enum value

    logger.info("Test with geo event_source, should succeed.")
    # Test with geo event_source (now implemented)
    filter_request = {
        "df_ids": [1, 2, 3],
        "event_source": "geo",
        "filter_operator": "intersection",
    }

    response = requests.post(
        f"{server_url}/api/filters/df_ids",
        json=filter_request,
        headers={"Content-Type": "application/json"},
    )

    # Geo filtering should now work (return 200)
    assert response.status_code == 200
    geo_result = response.json()
    assert geo_result["success"] is True
