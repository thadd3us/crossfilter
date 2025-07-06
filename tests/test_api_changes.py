"""Test API changes for row index support."""

import pytest
import requests
from pathlib import Path

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"


def test_session_status_with_data(server_with_data: str) -> None:
    """Test that session status includes the new fields."""
    server_url = server_with_data
    response = requests.get(f"{server_url}/api/session")
    assert response.status_code == 200

    data = response.json()
    assert data["has_data"] is True
    assert data["row_count"] == 100
    assert data["filtered_count"] == 100
    assert "columns" in data


def test_load_data_endpoint(server_with_data: str) -> None:
    """Test the load data endpoint with new request format."""
    server_url = server_with_data

    # Test loading data
    sample_path = str(Path(__file__).parent.parent / "test_data" / "sample_100.jsonl")
    response = requests.post(f"{server_url}/api/data/load", json={"file_path": sample_path})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Successfully loaded 100 records" in data["message"]


@pytest.mark.skip(reason="Not implemented")
def test_temporal_plot_endpoint(server_with_data: str) -> None:
    """Test temporal plot endpoint returns data with row indices."""
    server_url = server_with_data
    response = requests.get(f"{server_url}/api/plots/temporal?max_groups=1000")
    assert response.status_code == 200

    data = response.json()
    assert "plotly_plot" in data
    assert "data_type" in data
    assert "point_count" in data

    # Check that the plot data includes customdata with df_id
    plot_data = data["plotly_plot"]["data"][0]
    assert "customdata" in plot_data
    assert len(plot_data["customdata"]) > 0

    # Check that each customdata entry has df_id
    for custom_data_point in plot_data["customdata"]:
        assert "df_id" in custom_data_point
        assert isinstance(custom_data_point["df_id"], int)


@pytest.mark.skip(reason="Not implemented")
def test_apply_temporal_filter(server_with_data: str) -> None:
    """Test applying temporal filter with row indices."""
    server_url = server_with_data
    # Apply a filter with some row indices
    filter_request = {
        "row_indices": [0, 1, 2, 3, 4],  # Select first 5 rows
        "operation_type": "temporal",
        "description": "Test temporal filter",
    }

    response = requests.post(f"{server_url}/api/filters/apply", json=filter_request)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "filter_state" in data

    # Check that the filter was applied
    filter_state = data["filter_state"]
    assert filter_state["filtered_count"] == 5
    assert filter_state["total_count"] == 100


def test_reset_filters(server_with_data: str) -> None:
    """Test resetting filters."""
    server_url = server_with_data
    # First apply a filter
    filter_request = {
        "row_indices": [0, 1, 2],
        "operation_type": "temporal",
        "description": "Test filter to reset",
    }
    requests.post(f"{server_url}/api/filters/apply", json=filter_request)

    # Now reset
    response = requests.post(f"{server_url}/api/filters/reset")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True

    # Check session status shows all data visible
    status_response = requests.get(f"{server_url}/api/session")
    status_data = status_response.json()
    assert status_data["filtered_count"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
