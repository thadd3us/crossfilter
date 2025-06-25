"""Tests for session state management."""

import pytest
import pandas as pd

from crossfilter.core.session_state import SessionState
from crossfilter.core.schema_constants import SchemaColumns, DF_ID_COLUMN


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: [f"uuid_{i}" for i in range(20)],
            SchemaColumns.GPS_LATITUDE: [37.7749 + i * 0.001 for i in range(20)],
            SchemaColumns.GPS_LONGITUDE: [-122.4194 + i * 0.001 for i in range(20)],
            SchemaColumns.TIMESTAMP_UTC: [
                f"2024-01-01T{10 + i // 4}:00:00Z" for i in range(20)
            ],
        }
    )
    # Convert timestamp to datetime
    df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(
        df[SchemaColumns.TIMESTAMP_UTC], utc=True
    )
    # Set stable df_id index
    df.index.name = DF_ID_COLUMN
    return df


def test_session_state_initialization() -> None:
    """Test SessionState initialization."""
    session = SessionState()

    assert not session.has_data()
    assert session.data.empty
    assert session.quantized_data.empty

    summary = session.get_summary()
    assert summary["status"] == "empty"
    assert summary["message"] == "No data loaded"


def test_load_dataframe(sample_df: pd.DataFrame) -> None:
    """Test loading a DataFrame into session state."""
    session = SessionState()
    session.load_dataframe(sample_df)

    assert session.has_data()
    assert len(session.data) == 20
    assert session.data.index.name == DF_ID_COLUMN

    # Should have quantized data with additional columns
    quantized = session.quantized_data
    assert len(quantized) == 20
    assert len(quantized.columns) > len(sample_df.columns)

    # Check that H3 and temporal columns were added
    h3_cols = [col for col in quantized.columns if col.startswith("QUANTIZED_H3_")]
    temporal_cols = [
        col for col in quantized.columns if col.startswith("QUANTIZED_TIMESTAMP_")
    ]
    assert len(h3_cols) > 0
    assert len(temporal_cols) > 0


# THAD: Lots of tests in this filedon't have type annotations on parameters, please fix.
def test_session_state_metadata(sample_df) -> None:
    """Test metadata tracking."""
    session = SessionState()
    session.load_dataframe(sample_df)

    metadata = session.metadata
    assert metadata["shape"] == (20, 4)  # Original columns only
    assert SchemaColumns.GPS_LATITUDE in metadata["columns"]
    assert SchemaColumns.TIMESTAMP_UTC in metadata["columns"]

    summary = session.get_summary()
    assert summary["status"] == "loaded"
    assert summary["shape"] == (20, 4)
    assert "memory_usage" in summary
    assert "filter_state" in summary


def test_filter_state_integration(sample_df) -> None:
    """Test integration with filter state."""
    session = SessionState()
    session.load_dataframe(sample_df)

    filter_state = session.filter_state
    assert filter_state.total_count == 20
    assert filter_state.filter_count == 20  # All points initially visible

    # Apply a filter
    filtered_df_ids = set(range(0, 10))  # First 10 points
    filter_state.apply_spatial_filter(filtered_df_ids, "Test spatial filter")

    assert filter_state.filter_count == 10
    assert filter_state.can_undo

    # Get filtered data
    filtered = session.get_filtered_data()
    assert len(filtered) == 10
    assert all(idx in range(0, 10) for idx in filtered.index)


def test_spatial_aggregation_individual_points(sample_df) -> None:
    """Test spatial aggregation when under threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request more groups than we have points
    result = session.get_spatial_aggregation(max_groups=100)

    # Should return individual points
    assert len(result) == 20
    assert SchemaColumns.GPS_LATITUDE in result.columns
    assert SchemaColumns.GPS_LONGITUDE in result.columns
    assert DF_ID_COLUMN in result.columns


def test_spatial_aggregation_aggregated(sample_df) -> None:
    """Test spatial aggregation when over threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request fewer groups than we have points
    result = session.get_spatial_aggregation(max_groups=5)

    # Should return aggregated data
    assert len(result) <= 5
    assert "lat" in result.columns
    assert "lon" in result.columns
    assert "count" in result.columns
    assert "df_ids" in result.columns

    # Total count should match original data
    assert result["count"].sum() == 20


def test_temporal_aggregation_individual_points(sample_df) -> None:
    """Test temporal aggregation when under threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request more groups than we have points
    result = session.get_temporal_aggregation(max_groups=100)

    # Should return individual points
    assert len(result) == 20
    assert SchemaColumns.TIMESTAMP_UTC in result.columns
    assert "cumulative_count" in result.columns
    assert DF_ID_COLUMN in result.columns

    # Should be sorted by timestamp
    assert result[SchemaColumns.TIMESTAMP_UTC].is_monotonic_increasing

    # Cumulative count should go from 1 to 20
    assert list(result["cumulative_count"]) == list(range(1, 21))


def test_temporal_aggregation_aggregated(sample_df) -> None:
    """Test temporal aggregation when over threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request fewer groups than we have points
    result = session.get_temporal_aggregation(max_groups=3)

    # Should return aggregated data
    assert len(result) <= 3
    assert "count" in result.columns
    assert "cumulative_count" in result.columns
    assert "df_ids" in result.columns

    # Total count should match original data
    assert result["count"].sum() == 20

    # Cumulative count should end at total
    assert result["cumulative_count"].iloc[-1] == 20


def test_clear_session_state(sample_df) -> None:
    """Test clearing session state."""
    session = SessionState()
    session.load_dataframe(sample_df)

    assert session.has_data()

    session.clear()

    assert not session.has_data()
    assert session.data.empty
    assert session.quantized_data.empty
    assert session.filter_state.total_count == 0

    summary = session.get_summary()
    assert summary["status"] == "empty"


def test_data_property_setter(sample_df) -> None:
    """Test that the data property setter works correctly."""
    session = SessionState()

    # Use property setter
    session.data = sample_df

    assert session.has_data()
    assert len(session.data) == 20
    assert len(session.quantized_data) > len(
        sample_df.columns
    )  # Should have quantized columns


def test_filter_integration_with_aggregation(sample_df) -> None:
    """Test that filtering affects aggregation results."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Get initial aggregation
    initial_spatial = session.get_spatial_aggregation(max_groups=100)
    initial_temporal = session.get_temporal_aggregation(max_groups=100)

    assert len(initial_spatial) == 20
    assert len(initial_temporal) == 20

    # Apply filter to reduce to 10 points
    filtered_df_ids = set(range(0, 10))
    session.filter_state.apply_spatial_filter(filtered_df_ids, "Test filter")

    # Get aggregation after filtering
    filtered_spatial = session.get_spatial_aggregation(max_groups=100)
    filtered_temporal = session.get_temporal_aggregation(max_groups=100)

    assert len(filtered_spatial) == 10
    assert len(filtered_temporal) == 10

    # All df_ids should be in the filtered set
    assert all(df_id in filtered_df_ids for df_id in filtered_spatial[DF_ID_COLUMN])
    assert all(df_id in filtered_df_ids for df_id in filtered_temporal[DF_ID_COLUMN])


def test_empty_dataframe_handling() -> None:
    """Test handling of empty DataFrames."""
    session = SessionState()
    empty_df = pd.DataFrame(
        columns=[
            SchemaColumns.GPS_LATITUDE,
            SchemaColumns.GPS_LONGITUDE,
            SchemaColumns.TIMESTAMP_UTC,
        ]
    )
    empty_df.index.name = DF_ID_COLUMN

    session.load_dataframe(empty_df)

    assert not session.has_data()  # Empty DataFrame

    # Aggregation should handle empty data gracefully
    spatial_result = session.get_spatial_aggregation(max_groups=100)
    temporal_result = session.get_temporal_aggregation(max_groups=100)

    assert len(spatial_result) == 0
    assert len(temporal_result) == 0
