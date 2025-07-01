"""Tests for visualization plots module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from crossfilter.core.data_schema import load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.plots import create_fallback_scatter_geo
from crossfilter.visualization.temporal_cdf_plot import create_temporal_cdf


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load sample data for testing."""
    sample_path = Path(__file__).parent.parent.parent / "test_data" / "sample_100.jsonl"
    return load_jsonl_to_dataframe(sample_path)


@pytest.fixture
def session_state_with_data(sample_data: pd.DataFrame) -> SessionState:
    """Create session state with sample data loaded."""
    session_state = SessionState()
    session_state.load_dataframe(sample_data)
    return session_state


def test_create_temporal_cdf_json_serializable(session_state_with_data: SessionState) -> None:
    """Test that temporal CDF plot is JSON serializable (catches numpy array issues)."""
    # Get temporal aggregation data
    temporal_data = session_state_with_data.get_temporal_aggregation(max_groups=1000)
    
    # Create the plot
    plot = create_temporal_cdf(temporal_data)
    
    # Verify basic structure
    assert "data" in plot
    assert "layout" in plot
    assert len(plot["data"]) > 0
    
    # Most importantly: verify it's JSON serializable
    # This would have caught the numpy array serialization issue
    try:
        json_str = json.dumps(plot)
        assert len(json_str) > 0
    except (TypeError, ValueError) as e:
        pytest.fail(f"Plot is not JSON serializable: {e}")


def test_create_temporal_cdf_with_empty_data() -> None:
    """Test temporal CDF creation with empty DataFrame."""
    empty_df = pd.DataFrame()
    plot = create_temporal_cdf(empty_df)
    
    # Should handle empty data gracefully
    assert "data" in plot
    assert "layout" in plot
    
    # Should be JSON serializable
    json.dumps(plot)


def test_create_temporal_cdf_customdata_structure(session_state_with_data: SessionState) -> None:
    """Test that customdata has correct structure for row selection."""
    temporal_data = session_state_with_data.get_temporal_aggregation(max_groups=1000)
    plot = create_temporal_cdf(temporal_data)
    
    # Check customdata structure
    trace_data = plot["data"][0]
    assert "customdata" in trace_data
    
    customdata = trace_data["customdata"]
    assert len(customdata) > 0
    
    # Check each customdata entry has df_id and it's an int
    for custom_item in customdata:
        assert "df_id" in custom_item
        assert isinstance(custom_item["df_id"], int)
        
    # Verify JSON serializable
    json.dumps(customdata)


def test_create_fallback_scatter_geo_json_serializable(session_state_with_data: SessionState) -> None:
    """Test that geographic scatter plot is JSON serializable."""
    spatial_data = session_state_with_data.get_spatial_aggregation(max_groups=1000)
    
    plot = create_fallback_scatter_geo(spatial_data)
    
    # Verify basic structure
    assert "data" in plot
    assert "layout" in plot
    
    # Verify JSON serializable
    json.dumps(plot)


def test_create_fallback_scatter_geo_with_empty_data() -> None:
    """Test geographic scatter plot with empty DataFrame."""
    empty_df = pd.DataFrame()
    plot = create_fallback_scatter_geo(empty_df)
    
    # Should handle empty data gracefully
    assert "data" in plot
    assert "layout" in plot
    
    # Should be JSON serializable
    json.dumps(plot)