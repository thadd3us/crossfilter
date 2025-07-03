"""Tests for projection state management."""

import pandas as pd
import pytest

from crossfilter.core.projection_state import ProjectionState
from crossfilter.core.schema import SchemaColumns as C


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    df = pd.DataFrame({
        C.UUID_STRING: [f"uuid_{i}" for i in range(20)],
        C.GPS_LATITUDE: [37.7749 + i * 0.001 for i in range(20)],
        C.GPS_LONGITUDE: [-122.4194 + i * 0.001 for i in range(20)],
        C.TIMESTAMP_UTC: [f"2024-01-01T{10 + i // 4}:00:00Z" for i in range(20)],
    })
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df.index.name = C.DF_ID
    return df


@pytest.fixture
def bucketed_data() -> pd.DataFrame:
    """Create sample bucketed data for testing."""
    df = pd.DataFrame({
        "bucket_column": ["bucket_a", "bucket_b", "bucket_c"],
        C.GPS_LATITUDE: [37.7749, 37.7759, 37.7769],
        C.GPS_LONGITUDE: [-122.4194, -122.4204, -122.4214],
        "count": [5, 8, 7],
    })
    df.index.name = C.DF_ID
    return df


def test_projection_state_initialization() -> None:
    """Test ProjectionState initialization."""
    projection = ProjectionState(max_rows=1000)

    assert projection.max_rows == 1000
    assert projection.projection_df.empty
    assert projection.current_bucketing_column is None


def test_apply_filter_event_individual_points(sample_data: pd.DataFrame) -> None:
    """Test applying filter event in individual points mode."""
    projection = ProjectionState(max_rows=100)

    # Set up individual points mode (no bucketing column)
    projection.projection_df = sample_data.copy()
    projection.current_bucketing_column = None

    # Select first 10 points
    selected_df_ids = set(range(0, 10))
    result = projection.apply_filter_event(selected_df_ids, sample_data)

    assert len(result) == 10
    assert all(idx in range(0, 10) for idx in result.index)


def test_apply_filter_event_empty_selection(sample_data: pd.DataFrame) -> None:
    """Test applying filter event with empty selection."""
    projection = ProjectionState(max_rows=100)
    projection.projection_df = sample_data.copy()

    result = projection.apply_filter_event(set(), sample_data)
    assert result.empty


def test_apply_filter_event_aggregated_mode(sample_data: pd.DataFrame, bucketed_data: pd.DataFrame) -> None:
    """Test applying filter event in aggregated mode."""
    projection = ProjectionState(max_rows=100)

    # Set up aggregated mode
    projection.projection_df = bucketed_data.copy()
    projection.current_bucketing_column = "bucket_column"

    # Add bucket column to sample data
    sample_data_with_buckets = sample_data.copy()
    sample_data_with_buckets["bucket_column"] = ["bucket_a"] * 5 + ["bucket_b"] * 8 + ["bucket_c"] * 7

    # Select first bucket (df_id 0 in the projection)
    selected_df_ids = {0}
    result = projection.apply_filter_event(selected_df_ids, sample_data_with_buckets)

    # Should return rows with bucket_a
    assert len(result) == 5
    assert all(result["bucket_column"] == "bucket_a")


def test_apply_filter_event_aggregated_mode_missing_column(sample_data: pd.DataFrame, bucketed_data: pd.DataFrame) -> None:
    """Test applying filter event when bucketing column is missing from filtered data."""
    projection = ProjectionState(max_rows=100)

    # Set up aggregated mode with non-existent column
    projection.projection_df = bucketed_data.copy()
    projection.current_bucketing_column = "nonexistent_column"

    # Try to apply filter
    selected_df_ids = {0}
    result = projection.apply_filter_event(selected_df_ids, sample_data)

    # Should return original data (fallback behavior)
    assert len(result) == len(sample_data)


def test_apply_filter_event_invalid_df_ids(sample_data: pd.DataFrame, bucketed_data: pd.DataFrame) -> None:
    """Test applying filter event with invalid df_ids."""
    projection = ProjectionState(max_rows=100)

    # Set up aggregated mode
    projection.projection_df = bucketed_data.copy()  # Only 3 rows
    projection.current_bucketing_column = "bucket_column"

    # Add bucket column to sample data
    sample_data_with_buckets = sample_data.copy()
    sample_data_with_buckets["bucket_column"] = ["bucket_a"] * 20

    # Use invalid df_ids (beyond projection length)
    invalid_ids = {100, 200}
    result = projection.apply_filter_event(invalid_ids, sample_data_with_buckets)

    # Should return empty DataFrame for invalid selections
    assert result.empty


def test_apply_filter_event_aggregated_error_handling(sample_data: pd.DataFrame) -> None:
    """Test error handling in aggregated mode."""
    projection = ProjectionState(max_rows=100)

    # Set up problematic state (projection_df is empty but bucketing column is set)
    projection.projection_df = pd.DataFrame()
    projection.current_bucketing_column = "bucket_column"

    selected_df_ids = {0}
    result = projection.apply_filter_event(selected_df_ids, sample_data)

    # Should return original filtered_rows due to graceful fallback when bucketing column is missing
    assert len(result) == len(sample_data)


def test_get_summary() -> None:
    """Test getting projection summary."""
    projection = ProjectionState(max_rows=1000)

    summary = projection.get_summary()

    assert summary["max_rows"] == 1000
    assert summary["projection_rows"] == 0
    assert summary["current_bucketing_column"] is None
    assert summary["is_aggregated"] is False


def test_get_summary_with_data(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary with data."""
    projection = ProjectionState(max_rows=500)
    projection.projection_df = sample_data.copy()
    projection.current_bucketing_column = "test_column"

    summary = projection.get_summary()

    assert summary["max_rows"] == 500
    assert summary["projection_rows"] == 20
    assert summary["current_bucketing_column"] == "test_column"
    assert summary["is_aggregated"] is True


def test_max_rows_property() -> None:
    """Test max_rows property access and modification."""
    projection = ProjectionState(max_rows=100)

    assert projection.max_rows == 100

    projection.max_rows = 500
    assert projection.max_rows == 500


def test_projection_df_property(sample_data: pd.DataFrame) -> None:
    """Test projection_df property access."""
    projection = ProjectionState(max_rows=100)

    # Initially empty
    assert projection.projection_df.empty

    # Set some data
    projection.projection_df = sample_data.copy()
    retrieved = projection.projection_df

    # Should be a copy (not the same object)
    assert len(retrieved) == len(sample_data)
    assert retrieved is not projection.projection_df
