"""Test API changes for row index support."""

import json
from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from crossfilter.main import app, get_session_state
from crossfilter.core.data_schema import load_jsonl_to_dataframe


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    sample_path = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    return load_jsonl_to_dataframe(sample_path)


@pytest.fixture
def client_with_data(sample_data):
    """Create test client with sample data loaded."""
    session_state = get_session_state()
    session_state.load_dataframe(sample_data)
    return TestClient(app)


def test_session_status_with_data(client_with_data):
    """Test that session status includes the new fields."""
    response = client_with_data.get("/api/session")
    assert response.status_code == 200
    
    data = response.json()
    assert data["has_data"] is True
    assert data["row_count"] == 100
    assert data["filtered_count"] == 100
    assert "columns" in data


def test_load_data_endpoint(sample_data):
    """Test the load data endpoint with new request format."""
    client = TestClient(app)
    
    # Test loading data
    sample_path = str(Path(__file__).parent.parent / "test_data" / "sample_100.jsonl")
    response = client.post("/api/data/load", json={"file_path": sample_path})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Successfully loaded 100 records" in data["message"]


def test_temporal_plot_endpoint(client_with_data):
    """Test temporal plot endpoint returns data with row indices."""
    response = client_with_data.get("/api/plots/temporal?max_groups=1000")
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


def test_apply_temporal_filter(client_with_data):
    """Test applying temporal filter with row indices."""
    # Apply a filter with some row indices
    filter_request = {
        "row_indices": [0, 1, 2, 3, 4],  # Select first 5 rows
        "operation_type": "temporal",
        "description": "Test temporal filter"
    }
    
    response = client_with_data.post("/api/filters/apply", json=filter_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "filter_state" in data
    
    # Check that the filter was applied
    filter_state = data["filter_state"]
    assert filter_state["filtered_count"] == 5
    assert filter_state["total_count"] == 100


def test_reset_filters(client_with_data):
    """Test resetting filters."""
    # First apply a filter
    filter_request = {
        "row_indices": [0, 1, 2],
        "operation_type": "temporal", 
        "description": "Test filter to reset"
    }
    client_with_data.post("/api/filters/apply", json=filter_request)
    
    # Now reset
    response = client_with_data.post("/api/filters/reset")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    
    # Check session status shows all data visible
    status_response = client_with_data.get("/api/session")
    status_data = status_response.json()
    assert status_data["filtered_count"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])