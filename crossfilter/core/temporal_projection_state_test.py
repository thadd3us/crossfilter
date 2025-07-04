"""Tests for temporal projection state management."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.schema import (
    SchemaColumns as C,
    TemporalLevel,
    get_temporal_column_name,
)
from crossfilter.core.temporal_projection_state import TemporalProjectionState


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data with temporal information."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(20)],
            C.GPS_LATITUDE: [37.7749 + i * 0.001 for i in range(20)],
            C.GPS_LONGITUDE: [-122.4194 + i * 0.001 for i in range(20)],
            C.TIMESTAMP_UTC: [f"2024-01-01T{10 + i // 4}:00:00Z" for i in range(20)],
        }
    )
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df.index.name = C.DF_ID

    # Add bucketed columns
    return add_bucketed_columns(df)


def test_temporal_projection_initialization() -> None:
    """Test TemporalProjectionState initialization."""
    projection = TemporalProjectionState(max_rows=1000)

    assert projection.projection_state.max_rows == 1000
    assert projection.projection_state.projection_df.empty
    assert projection.current_aggregation_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_update_projection_individual_points(
    sample_data: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test updating projection with individual points (under threshold)."""
    projection = TemporalProjectionState(max_rows=100)
    projection.update_projection(sample_data)
    assert projection.current_aggregation_level is None
    assert projection.projection_state.current_bucketing_column is None
    result = projection.projection_state.projection_df
    assert result.to_dict(orient="records") == snapshot


def test_update_projection_aggregated(
    sample_data: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test updating projection with aggregation (over threshold)."""
    projection = TemporalProjectionState(max_rows=5)
    projection.update_projection(sample_data)

    result = projection.projection_state.projection_df

    assert projection.current_aggregation_level is TemporalLevel.SECOND
    assert (
        projection.projection_state.current_bucketing_column
        == get_temporal_column_name(TemporalLevel.SECOND)
    )
    result = projection.projection_state.projection_df
    assert result.to_dict(orient="records") == snapshot


def test_update_projection_empty_data() -> None:
    """Test updating projection with empty data."""
    projection = TemporalProjectionState(max_rows=100)
    empty_df = pd.DataFrame()

    projection.update_projection(empty_df)

    assert projection.projection_state.projection_df.empty
    assert projection.current_aggregation_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_update_projection_no_timestamp(sample_data: pd.DataFrame) -> None:
    """Test updating projection with data missing timestamp column."""
    projection = TemporalProjectionState(max_rows=1)

    # Remove timestamp column
    data_without_timestamp = sample_data[
        [C.UUID_STRING, C.GPS_LATITUDE, C.GPS_LONGITUDE]
    ]
    with pytest.raises(AssertionError, match="QUANTIZED_TIMESTAMP_SECOND"):
        projection.update_projection(data_without_timestamp)

    assert projection.projection_state.projection_df.empty
    assert projection.current_aggregation_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_apply_filter_event_individual_points(sample_data: pd.DataFrame) -> None:
    """Test applying filter event in individual points mode."""
    projection = TemporalProjectionState(max_rows=100)
    projection.update_projection(sample_data)

    # Select first 10 points
    selected_df_ids = set(range(0, 10))
    result = projection.apply_filter_event(selected_df_ids, sample_data)

    assert len(result) == 10
    assert all(idx in range(0, 10) for idx in result.index)


def test_apply_filter_event_aggregated(sample_data: pd.DataFrame) -> None:
    """Test applying filter event in aggregated mode."""
    projection = TemporalProjectionState(max_rows=5)
    projection.update_projection(sample_data)

    # Should be in aggregated mode
    assert projection.current_aggregation_level is not None

    # Select first bucket (df_id 0 in the projection)
    selected_df_ids = {0}
    result = projection.apply_filter_event(selected_df_ids, sample_data)

    # Should return some subset of original data
    assert len(result) > 0
    assert len(result) <= 20


def test_apply_filter_event_empty_selection(sample_data: pd.DataFrame) -> None:
    """Test applying filter event with empty selection."""
    projection = TemporalProjectionState(max_rows=100)
    projection.update_projection(sample_data)

    result = projection.apply_filter_event(set(), sample_data)
    assert result.empty


def test_apply_filter_event_invalid_ids(sample_data: pd.DataFrame) -> None:
    """Test applying filter event with invalid df_ids."""
    projection = TemporalProjectionState(max_rows=5)  # Force aggregation
    projection.update_projection(sample_data)

    # Use invalid df_ids (beyond projection length)
    invalid_ids = {100, 200}
    result = projection.apply_filter_event(invalid_ids, sample_data)

    # Should return empty DataFrame for invalid selections
    assert result.empty


def test_get_summary(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary."""
    projection = TemporalProjectionState(max_rows=100)
    projection.update_projection(sample_data)

    summary = projection.get_summary()

    assert summary["max_rows"] == 100
    assert summary["projection_rows"] == 20
    assert summary["aggregation_level"] is None  # Individual points
    assert summary["target_column"] is None
    assert summary["is_aggregated"] is False


def test_get_summary_aggregated(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary for aggregated data."""
    projection = TemporalProjectionState(max_rows=5)
    projection.update_projection(sample_data)

    summary = projection.get_summary()

    assert summary["max_rows"] == 5
    assert summary["projection_rows"] <= 5
    assert summary["aggregation_level"] is not None
    assert summary["target_column"] is not None
    assert summary["is_aggregated"] is True


def test_max_rows_threshold_boundary(sample_data: pd.DataFrame) -> None:
    """Test behavior at exact max_rows threshold."""
    # Use exactly 20 max_rows (same as data size)
    projection = TemporalProjectionState(max_rows=20)
    projection.update_projection(sample_data)

    # Should show individual points
    result = projection.projection_state.projection_df
    assert len(result) == 20
    assert C.COUNT not in result.columns
    assert projection.current_aggregation_level is None

    # Now use 19 max_rows (one less than data size)
    projection = TemporalProjectionState(max_rows=19)
    projection.update_projection(sample_data)

    # Should aggregate
    result = projection.projection_state.projection_df
    assert len(result) <= 19
    assert C.COUNT in result.columns
    assert projection.current_aggregation_level is not None
