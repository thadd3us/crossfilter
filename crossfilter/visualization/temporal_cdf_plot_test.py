"""Tests for temporal CDF plot module."""

import json
from pathlib import Path

import pandas as pd
import plotly
import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension

from crossfilter.core.data_schema import load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.temporal_cdf_plot import create_temporal_cdf


class HTMLSnapshotExtension(SingleFileSnapshotExtension):
    """Custom syrupy extension to save HTML files with .html extension."""
    _file_extension = "html"
    
    def serialize(self, data, **kwargs):
        """Serialize string data to bytes for file storage."""
        if isinstance(data, str):
            return data.encode('utf-8')
        return super().serialize(data, **kwargs)


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


@pytest.fixture
def aggregated_temporal_data(session_state_with_data: SessionState) -> pd.DataFrame:
    """Get aggregated temporal data for testing."""
    return session_state_with_data.get_temporal_aggregation(max_groups=1000)


@pytest.fixture 
def individual_temporal_data(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Create individual temporal data for testing."""
    # Use a subset of sample data as individual points
    individual_df = sample_data.head(50).copy()
    individual_df = individual_df.reset_index(drop=True)
    return individual_df


def test_temporal_cdf_aggregated_data_snapshot(snapshot, aggregated_temporal_data: pd.DataFrame) -> None:
    """Test temporal CDF plot with aggregated data using syrupy snapshots."""
    plot = create_temporal_cdf(aggregated_temporal_data, title="Test Aggregated CDF")
    
    # Verify basic structure
    assert "data" in plot
    assert "layout" in plot
    assert len(plot["data"]) > 0
    
    # Test JSON serialization
    json.dumps(plot)
    
    # Create HTML content for snapshot with deterministic ID
    fig = plotly.graph_objects.Figure(plot)
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        div_id='test-plot-div'  # Use deterministic ID instead of random
    )
    
    # Snapshot test - saves as HTML file managed by syrupy
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
    


def test_temporal_cdf_individual_data_snapshot(snapshot, individual_temporal_data: pd.DataFrame) -> None:
    """Test temporal CDF plot with individual data using syrupy snapshots."""
    plot = create_temporal_cdf(individual_temporal_data, title="Test Individual CDF")
    
    # Verify basic structure
    assert "data" in plot
    assert "layout" in plot
    assert len(plot["data"]) > 0
    
    # Test JSON serialization
    json.dumps(plot)
    
    # Create HTML content for snapshot with deterministic ID
    fig = plotly.graph_objects.Figure(plot)
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        div_id='test-plot-div'  # Use deterministic ID instead of random
    )
    
    # Snapshot test - saves as HTML file managed by syrupy
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
    


def test_temporal_cdf_empty_data_snapshot(snapshot) -> None:
    """Test temporal CDF plot with empty data using syrupy snapshots."""
    empty_df = pd.DataFrame()
    plot = create_temporal_cdf(empty_df, title="Test Empty CDF")
    
    # Should handle empty data gracefully
    assert "data" in plot
    assert "layout" in plot
    
    # Should be JSON serializable
    json.dumps(plot)
    
    # Create HTML content for snapshot with deterministic ID
    fig = plotly.graph_objects.Figure(plot)
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        div_id='test-plot-div'  # Use deterministic ID instead of random
    )
    
    # Snapshot test - saves as HTML file managed by syrupy
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
    


def test_temporal_cdf_no_timestamp_data_snapshot(snapshot) -> None:
    """Test temporal CDF plot with data but no timestamp columns."""
    df_without_timestamp = pd.DataFrame({
        'some_column': [1, 2, 3],
        'another_column': ['a', 'b', 'c']
    })
    plot = create_temporal_cdf(df_without_timestamp, title="Test No Timestamp CDF")
    
    # Should handle missing timestamp gracefully
    assert "data" in plot
    assert "layout" in plot
    
    # Should be JSON serializable
    json.dumps(plot)
    
    # Create HTML content for snapshot with deterministic ID
    fig = plotly.graph_objects.Figure(plot)
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        div_id='test-plot-div'  # Use deterministic ID instead of random
    )
    
    # Snapshot test - saves as HTML file managed by syrupy
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
    


def test_temporal_cdf_customdata_structure(aggregated_temporal_data: pd.DataFrame) -> None:
    """Test that customdata has correct structure for row selection."""
    plot = create_temporal_cdf(aggregated_temporal_data)
    
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


def test_temporal_cdf_df_id_consistency_aggregated(aggregated_temporal_data: pd.DataFrame) -> None:
    """Test that df_id values in customdata reference the original DataFrame index."""
    plot = create_temporal_cdf(aggregated_temporal_data)
    
    trace_data = plot["data"][0]
    customdata = trace_data["customdata"]
    
    # Extract all df_ids from customdata
    df_ids_in_plot = {item["df_id"] for item in customdata}
    
    # Should reference indices from the original aggregated DataFrame
    original_indices = set(aggregated_temporal_data.index.astype(int))
    
    # All df_ids in plot should be from original indices
    assert df_ids_in_plot.issubset(original_indices)


def test_temporal_cdf_df_id_consistency_individual(individual_temporal_data: pd.DataFrame) -> None:
    """Test that df_id values match DataFrame index for individual data."""
    plot = create_temporal_cdf(individual_temporal_data)
    
    trace_data = plot["data"][0]
    customdata = trace_data["customdata"]
    
    # Extract all df_ids from customdata
    df_ids_in_plot = [item["df_id"] for item in customdata]
    
    # For individual data, should match the DataFrame index exactly
    expected_indices = list(individual_temporal_data.index.astype(int))
    
    assert df_ids_in_plot == expected_indices


def test_temporal_cdf_json_serialization(aggregated_temporal_data: pd.DataFrame) -> None:
    """Test that temporal CDF plot is fully JSON serializable."""
    plot = create_temporal_cdf(aggregated_temporal_data)
    
    # Should not raise any exceptions
    json_str = json.dumps(plot)
    assert len(json_str) > 0
    
    # Should be able to round-trip
    parsed = json.loads(json_str)
    assert parsed == plot