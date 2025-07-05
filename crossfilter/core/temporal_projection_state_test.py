"""Tests for temporal projection state management."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    FilterOperatorType,
    ProjectionType,
)
from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.schema import (
    SchemaColumns as C,
    DataType,
)
from crossfilter.core.schema import (
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
            C.DATA_TYPE: [
                [DataType.GPX_WAYPOINT, DataType.PHOTO][i % 2] for i in range(20)
            ],
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


def test_update_projection_empty_data(
    sample_data: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test updating projection with aggregation (over threshold)."""
    sample_data = sample_data.iloc[:0]
    projection = TemporalProjectionState(max_rows=5)
    projection.update_projection(sample_data)
    assert projection.current_aggregation_level is None
    result = projection.projection_state.projection_df
    assert result.empty


def test_get_summary(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary."""
    projection = TemporalProjectionState(max_rows=100)
    projection.update_projection(sample_data)

    summary = projection.get_summary()

    assert summary["max_rows"] == 100
    assert summary["projection_rows"] == 20
    assert summary["aggregation_level"] is None  # Individual points
    assert summary["current_bucketing_column"] is None
    assert summary["is_aggregated"] is False


def test_get_summary_aggregated(sample_data: pd.DataFrame) -> None:
    """Test getting projection summary for aggregated data."""
    projection = TemporalProjectionState(max_rows=5)
    projection.projection_state.groupby_column = None
    projection.update_projection(sample_data)

    summary = projection.get_summary()

    assert summary["max_rows"] == 5
    assert summary["projection_rows"] <= 5
    assert summary["aggregation_level"] is not None
    assert summary["current_bucketing_column"] is not None
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
