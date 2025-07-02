"""Tests for visualization plots module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from crossfilter.core.schema import load_jsonl_to_dataframe
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


def test_create_fallback_scatter_geo_json_serializable(
    session_state_with_data: SessionState,
) -> None:
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
