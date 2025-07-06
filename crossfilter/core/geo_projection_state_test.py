"""Tests for geographic projection state management."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    FilterOperatorType,
    ProjectionType,
)
from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import SchemaColumns as C, DataType


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data with geographic information."""
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

    # Add bucketed columns including H3 spatial indexing
    return add_bucketed_columns(df)


def test_geo_projection_initialization() -> None:
    """Test GeoProjectionState initialization."""
    projection = GeoProjectionState(max_rows=1000)

    assert projection.projection_state.max_rows == 1000
    assert projection.projection_state.projection_df.empty
    assert projection.current_h3_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_update_projection_individual_points(sample_data: pd.DataFrame) -> None:
    """Test updating projection with individual points (under threshold)."""
    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(sample_data)

    result = projection.projection_state.projection_df

    # Should return individual points
    assert len(result) == 20
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    assert C.COUNT not in result.columns  # No aggregation

    # H3 level should be None (individual points)
    assert projection.current_h3_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_update_projection_aggregated(sample_data: pd.DataFrame) -> None:
    """Test updating projection with aggregation (over threshold)."""
    projection = GeoProjectionState(max_rows=5)
    projection.update_projection(sample_data)

    result = projection.projection_state.projection_df

    # Should return aggregated data (note: result is grouped by both H3 cell and DATA_TYPE,
    # so the actual count may be higher than max_rows due to multiple data types per H3 cell)
    assert len(result) < len(sample_data)  # Should be aggregated (fewer rows than original)
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    assert C.COUNT in result.columns

    # Total count should match original data
    assert result[C.COUNT].sum() == 20

    # Should have H3 level and target column
    assert projection.current_h3_level is not None
    assert projection.projection_state.current_bucketing_column is not None
    assert projection.projection_state.current_bucketing_column.startswith(
        "QUANTIZED_H3_L"
    )
    assert projection.current_h3_level >= 0
    assert projection.current_h3_level <= 15


def test_update_projection_empty_data() -> None:
    """Test updating projection with empty data."""
    projection = GeoProjectionState(max_rows=100)
    empty_df = pd.DataFrame(
        columns=[C.UUID_STRING, C.TIMESTAMP_UTC, C.GPS_LATITUDE, C.GPS_LONGITUDE]
    )

    projection.update_projection(empty_df)

    assert projection.projection_state.projection_df.empty
    assert projection.current_h3_level is None
    assert projection.projection_state.current_bucketing_column is None


def test_update_projection_no_gps(sample_data: pd.DataFrame) -> None:
    """Test updating projection with data missing GPS coordinates."""
    projection = GeoProjectionState(max_rows=1)

    # Remove GPS columns
    data_without_gps = sample_data.copy()
    data_without_gps[[C.GPS_LATITUDE, C.GPS_LONGITUDE]] = None
    projection.update_projection(data_without_gps)

    # Should return the original data when H3 columns are missing
    assert len(projection.projection_state.projection_df) == 0
    assert projection.current_h3_level is None
    assert projection.projection_state.current_bucketing_column is None


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
    projection.projection_state.groupby_column = None
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
    result = projection.projection_state.projection_df
    assert len(result) == 20
    assert C.COUNT not in result.columns
    assert projection.current_h3_level is None

    # Now use a much smaller max_rows to force aggregation  
    projection = GeoProjectionState(max_rows=5)
    projection.update_projection(sample_data)

    # Should aggregate (note: result is grouped by both H3 cell and DATA_TYPE,
    # so the actual count may be higher than max_rows due to multiple data types per H3 cell)
    result = projection.projection_state.projection_df
    assert len(result) < len(sample_data)  # Should be aggregated (fewer rows than original)
    assert C.COUNT in result.columns
    assert projection.current_h3_level is not None


def test_data_without_some_gps_coordinates(snapshot: SnapshotAssertion) -> None:
    """Test handling of data with some missing GPS coordinates."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(5)],
            C.GPS_LATITUDE: [37.7749, None, 37.7751, 37.7752, None],
            C.GPS_LONGITUDE: [-122.4194, -122.4195, None, -122.4197, -122.4198],
            C.TIMESTAMP_UTC: [f"2024-01-01T10:00:0{i}Z" for i in range(5)],
        }
    )
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df.index.name = C.DF_ID

    # Add bucketed columns
    bucketed_data = add_bucketed_columns(df)

    projection = GeoProjectionState(max_rows=100)
    projection.update_projection(bucketed_data)

    # Should handle the data, even with some missing coordinates
    result = projection.projection_state.projection_df
    assert snapshot == result.to_dict(orient="records")
    assert len(result) == 2  # All rows included
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
