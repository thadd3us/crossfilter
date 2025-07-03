"""Tests for geographic projection state management."""

import pandas as pd
import pytest

from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import SchemaColumns as C


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data with geographic information."""
    df = pd.DataFrame({
        C.UUID_STRING: [f"uuid_{i}" for i in range(20)],
        C.GPS_LATITUDE: [37.7749 + i * 0.001 for i in range(20)],
        C.GPS_LONGITUDE: [-122.4194 + i * 0.001 for i in range(20)],
        C.TIMESTAMP_UTC: [f"2024-01-01T{10 + i // 4}:00:00Z" for i in range(20)],
    })
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df.index.name = C.DF_ID
    
    # Add bucketed columns including H3 spatial indexing
    return add_bucketed_columns(df)


def test_geo_projection_initialization() -> None:
    """Test GeoProjectionState initialization."""
    projection = GeoProjectionState(max_rows=1000)
    
    assert projection.max_rows == 1000
    assert projection.projection_df.empty
    assert projection.current_h3_level is None
    assert projection.current_target_column is None


def test_update_projection_individual_points(sample_data: pd.DataFrame) -> None:
    """Test updating projection with individual points (under threshold)."""
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(sample_data)
    
    result = projection.projection_df
    
    # Should return individual points
    assert len(result) == 20
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    assert C.DF_ID in result.columns
    assert "count" not in result.columns  # No aggregation
    
    # H3 level should be None (individual points)
    assert projection.current_h3_level is None
    assert projection.current_target_column is None


def test_update_projection_aggregated(sample_data: pd.DataFrame) -> None:
    """Test updating projection with aggregation (over threshold)."""
    projection = GeoProjectionState(max_rows=5)
    projection.update_projection(sample_data)
    
    result = projection.projection_df
    
    # Should return aggregated data
    assert len(result) <= 5
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    assert "count" in result.columns
    
    # Total count should match original data
    assert result["count"].sum() == 20
    
    # Should have H3 level and target column
    assert projection.current_h3_level is not None
    assert projection.current_target_column is not None
    assert projection.current_target_column.startswith("QUANTIZED_H3_L")
    assert projection.current_h3_level >= 0
    assert projection.current_h3_level <= 15


def test_update_projection_empty_data() -> None:
    """Test updating projection with empty data."""
    projection = GeoProjectionState(max_rows=100)
    empty_df = pd.DataFrame()
    
    projection.update_projection(empty_df)
    
    assert projection.projection_df.empty
    assert projection.current_h3_level is None
    assert projection.current_target_column is None


def test_update_projection_no_gps(sample_data: pd.DataFrame) -> None:
    """Test updating projection with data missing GPS coordinates."""
    projection = GeoProjectionState(max_rows=100)
    
    # Remove GPS columns
    data_without_gps = sample_data.drop(columns=[C.GPS_LATITUDE, C.GPS_LONGITUDE])
    projection.update_projection(data_without_gps)
    
    assert projection.projection_df.empty
    assert projection.current_h3_level is None
    assert projection.current_target_column is None


def test_apply_filter_event_individual_points(sample_data: pd.DataFrame) -> None:
    """Test applying filter event in individual points mode."""
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(sample_data)
    
    # Select first 10 points
    selected_df_ids = set(range(0, 10))
    result = projection.apply_filter_event(selected_df_ids, sample_data)
    
    assert len(result) == 10
    assert all(idx in range(0, 10) for idx in result.index)


def test_apply_filter_event_aggregated(sample_data: pd.DataFrame) -> None:
    """Test applying filter event in aggregated mode."""
    projection = GeoProjectionState(max_rows=5)
    projection.update_projection(sample_data)
    
    # Should be in aggregated mode
    assert projection.current_h3_level is not None
    
    # Select first bucket (df_id 0 in the projection)
    selected_df_ids = {0}
    result = projection.apply_filter_event(selected_df_ids, sample_data)
    
    # Should return some subset of original data
    assert len(result) > 0
    assert len(result) <= 20


def test_apply_filter_event_empty_selection(sample_data: pd.DataFrame) -> None:
    """Test applying filter event with empty selection."""
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(sample_data)
    
    result = projection.apply_filter_event(set(), sample_data)
    assert result.empty


def test_apply_filter_event_invalid_ids(sample_data: pd.DataFrame) -> None:
    """Test applying filter event with invalid df_ids."""
    projection = GeoProjectionState(max_rows=5)  # Force aggregation
    projection.update_projection(sample_data)
    
    # Use invalid df_ids (beyond projection length)
    invalid_ids = {100, 200}
    result = projection.apply_filter_event(invalid_ids, sample_data)
    
    # Should return empty DataFrame for invalid selections
    assert result.empty


def test_get_summary(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary."""
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(sample_data)
    
    summary = projection.get_summary()
    
    assert summary["max_rows"] == 100
    assert summary["projection_rows"] == 20
    assert summary["h3_level"] is None  # Individual points
    assert summary["target_column"] is None
    assert summary["is_aggregated"] is False


def test_get_summary_aggregated(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary for aggregated data."""
    projection = GeoProjectionState(max_rows=5)
    projection.update_projection(sample_data)
    
    summary = projection.get_summary()
    
    assert summary["max_rows"] == 5
    assert summary["projection_rows"] <= 5
    assert summary["h3_level"] is not None
    assert summary["target_column"] is not None
    assert summary["is_aggregated"] is True


def test_max_rows_threshold_boundary(sample_data: pd.DataFrame) -> None:
    """Test behavior at exact max_rows threshold."""
    # Use exactly 20 max_rows (same as data size)
    projection = GeoProjectionState(max_rows=20)
    projection.update_projection(sample_data)
    
    # Should show individual points
    result = projection.projection_df
    assert len(result) == 20
    assert "count" not in result.columns
    assert projection.current_h3_level is None
    
    # Now use 19 max_rows (one less than data size)
    projection = GeoProjectionState(max_rows=19)
    projection.update_projection(sample_data)
    
    # Should aggregate
    result = projection.projection_df
    assert len(result) <= 19
    assert "count" in result.columns
    assert projection.current_h3_level is not None


def test_data_without_some_gps_coordinates() -> None:
    """Test handling of data with some missing GPS coordinates."""
    df = pd.DataFrame({
        C.UUID_STRING: [f"uuid_{i}" for i in range(5)],
        C.GPS_LATITUDE: [37.7749, None, 37.7751, 37.7752, None],
        C.GPS_LONGITUDE: [-122.4194, -122.4195, None, -122.4197, -122.4198],
        C.TIMESTAMP_UTC: [f"2024-01-01T10:00:0{i}Z" for i in range(5)],
    })
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df.index.name = C.DF_ID
    
    # Add bucketed columns
    bucketed_data = add_bucketed_columns(df)
    
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(bucketed_data)
    
    # Should handle the data, even with some missing coordinates
    result = projection.projection_df
    assert len(result) == 5  # All rows included
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns