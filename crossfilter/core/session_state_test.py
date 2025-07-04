"""Tests for session state management."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.schema import FilterEvent, ProjectionType
from crossfilter.core.schema import SchemaColumns as C
from crossfilter.core.session_state import SessionState


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(20)],
            C.GPS_LATITUDE: [37.7749 + i * 0.001 for i in range(20)],
            C.GPS_LONGITUDE: [-122.4194 + i * 0.001 for i in range(20)],
            C.TIMESTAMP_UTC: [f"2024-01-01T{10 + i // 4}:00:00Z" for i in range(20)],
        }
    )
    # Convert timestamp to datetime
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    # Set stable df_id index
    df.index.name = C.DF_ID
    return df


def test_session_state_initialization() -> None:
    """Test SessionState initialization."""
    session = SessionState()

    assert len(session.all_rows) == 0
    assert session.all_rows.empty
    assert session.filtered_rows.empty

    summary = session.get_summary()
    assert summary["all_rows_count"] == 0
    assert summary["filtered_rows_count"] == 0


def test_load_dataframe(sample_df: pd.DataFrame) -> None:
    """Test loading a DataFrame into session state."""
    session = SessionState()
    session.load_dataframe(sample_df)

    assert len(session.all_rows) > 0
    assert len(session.all_rows) == 20
    assert len(session.filtered_rows) == 20
    assert session.all_rows.index.name == C.DF_ID
    assert session.filtered_rows.index.name == C.DF_ID

    # Check that bucketed columns were added
    all_rows = session.all_rows
    assert any(col.startswith("QUANTIZED_H3_L") for col in all_rows.columns)
    assert any(col.startswith("QUANTIZED_TIMESTAMP_") for col in all_rows.columns)


def test_session_state_metadata(sample_df: pd.DataFrame) -> None:
    """Test metadata tracking."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Check data directly from DataFrame
    assert len(session.all_rows) == 20  # Same number of rows
    assert len(session.all_rows.columns) > 4  # More columns due to bucketing
    assert C.GPS_LATITUDE in session.all_rows.columns
    assert C.TIMESTAMP_UTC in session.all_rows.columns

    summary = session.get_summary()
    assert summary["all_rows_count"] == 20
    assert summary["filtered_rows_count"] == 20
    assert "memory_usage" in summary
    assert "temporal_projection" in summary
    assert "geo_projection" in summary


def test_projection_states(sample_df: pd.DataFrame) -> None:
    """Test that projection states are properly initialized and updated."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Check temporal projection
    temporal_proj = session.temporal_projection
    assert temporal_proj.projection_state.max_rows == 100000  # default
    assert len(temporal_proj.projection_state.projection_df) == 20
    assert temporal_proj.current_aggregation_level is None  # individual points

    # Check geo projection
    geo_proj = session.geo_projection
    assert geo_proj.projection_state.max_rows == 100000  # default
    assert len(geo_proj.projection_state.projection_df) == 20
    assert geo_proj.current_h3_level is None  # individual points


def test_spatial_aggregation_individual_points(sample_df: pd.DataFrame) -> None:
    """Test spatial aggregation when under threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request more groups than we have points
    result = session.get_geo_aggregation()

    # Should return individual points
    assert len(result) == 20
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    # Should not have COUNT column for individual points
    assert C.COUNT not in result.columns


def test_spatial_aggregation_aggregated(sample_df: pd.DataFrame) -> None:
    """Test spatial aggregation when over threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request fewer groups than we have points
    session.geo_projection.projection_state.max_rows = 5
    session.geo_projection.update_projection(session.filtered_rows)
    result = session.get_geo_aggregation()

    # Should return aggregated data
    assert len(result) <= 5
    assert C.GPS_LATITUDE in result.columns
    assert C.GPS_LONGITUDE in result.columns
    assert C.COUNT in result.columns

    # Total count should match original data
    assert result[C.COUNT].sum() == 20


def test_temporal_aggregation_individual_points(sample_df: pd.DataFrame) -> None:
    """Test temporal aggregation when under threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Request more groups than we have points
    result = session.get_temporal_projection()

    # Should return individual points
    assert len(result) == 20
    assert result.index.name == C.DF_ID
    assert C.TIMESTAMP_UTC in result.columns
    assert C.COUNT not in result.columns

    # Should be sorted by timestamp
    assert result[C.TIMESTAMP_UTC].is_monotonic_increasing


def test_temporal_aggregation_aggregated(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test temporal aggregation when over threshold."""
    session = SessionState()
    session.load_dataframe(sample_df)

    session.temporal_projection.projection_state.max_rows = 3
    session.temporal_projection.update_projection(session.filtered_rows)
    result = session.get_temporal_projection()

    assert result.to_dict(orient="records") == snapshot


def test_apply_filter_event_temporal(sample_df: pd.DataFrame) -> None:
    """Test applying a temporal filter event."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Get initial counts
    assert len(session.filtered_rows) == 20

    # Apply a temporal filter (select first 10 points)
    selected_df_ids = set(range(0, 10))
    filter_event = FilterEvent(ProjectionType.TEMPORAL, selected_df_ids)
    session.apply_filter_event(filter_event)

    # Check that filtered_rows was updated
    assert len(session.filtered_rows) == 10
    assert all(idx in range(0, 10) for idx in session.filtered_rows.index)

    # Check that projections were updated
    temporal_result = session.get_temporal_projection()
    assert len(temporal_result) == 10


def test_apply_filter_event_geo(sample_df: pd.DataFrame) -> None:
    """Test applying a geographic filter event."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Get initial counts
    assert len(session.filtered_rows) == 20

    # Apply a geographic filter (select first 10 points)
    selected_df_ids = set(range(0, 10))
    filter_event = FilterEvent(ProjectionType.GEO, selected_df_ids)
    with pytest.raises(ValueError, match="Invalid projection type: geo"):
        session.apply_filter_event(filter_event)


def test_reset_filters(sample_df: pd.DataFrame) -> None:
    """Test resetting filters."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Apply a filter first
    selected_df_ids = set(range(0, 10))
    filter_event = FilterEvent(ProjectionType.TEMPORAL, selected_df_ids)
    session.apply_filter_event(filter_event)
    assert len(session.filtered_rows) == 10

    # Reset filters
    session.reset_filters()

    # Should be back to all data
    assert len(session.filtered_rows) == 20
    assert len(session.filtered_rows) == len(session.all_rows)


def test_clear_session_state(sample_df: pd.DataFrame) -> None:
    """Test clearing session state."""
    session = SessionState()
    session.load_dataframe(sample_df)

    assert len(session.all_rows) > 0

    session.clear()

    assert len(session.all_rows) == 0
    assert session.all_rows.empty
    assert session.filtered_rows.empty

    summary = session.get_summary()
    assert summary["all_rows_count"] == 0


def test_empty_dataframe_handling() -> None:
    """Test handling of empty DataFrames."""
    session = SessionState()
    empty_df = pd.DataFrame(
        columns=[
            C.GPS_LATITUDE,
            C.GPS_LONGITUDE,
            C.TIMESTAMP_UTC,
        ]
    )
    empty_df.index.name = C.DF_ID

    session.load_dataframe(empty_df)

    assert len(session.all_rows) == 0  # Empty DataFrame

    # Aggregation should handle empty data gracefully
    spatial_result = session.get_geo_aggregation()
    temporal_result = session.get_temporal_projection()

    assert len(spatial_result) == 0
    assert len(temporal_result) == 0


def test_filter_integration_with_aggregation(sample_df: pd.DataFrame) -> None:
    """Test that filtering affects aggregation results."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Get initial aggregation
    initial_spatial = session.get_geo_aggregation()
    initial_temporal = session.get_temporal_projection()

    assert len(initial_spatial) == 20
    assert len(initial_temporal) == 20

    # Apply filter to reduce to 10 points
    filtered_df_ids = set(range(5, 15))
    filter_event = FilterEvent(ProjectionType.TEMPORAL, filtered_df_ids)
    session.apply_filter_event(filter_event)

    # Get aggregation after filtering
    filtered_spatial = session.get_geo_aggregation()
    filtered_temporal = session.get_temporal_projection()

    assert filtered_spatial.index.to_list() == list(filtered_df_ids)
    assert filtered_temporal.index.to_list() == list(filtered_df_ids)


def test_projection_max_rows_updates(sample_df: pd.DataFrame) -> None:
    """Test that max_rows parameter updates projections correctly."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Initially should show individual points
    temporal_result = session.get_temporal_projection()
    assert C.COUNT not in temporal_result.columns  # Individual points

    # Change to force aggregation
    session.temporal_projection.projection_state.max_rows = 5
    session.temporal_projection.update_projection(session.filtered_rows)
    temporal_result = session.get_temporal_projection()
    assert C.COUNT in temporal_result.columns  # Aggregated

    # Verify projection state was updated
    assert session.temporal_projection.projection_state.max_rows == 5


def test_unknown_projection_type(sample_df: pd.DataFrame) -> None:
    """Test handling of unknown projection types."""
    session = SessionState()
    session.load_dataframe(sample_df)

    # Try to apply filter with unknown projection type
    filter_event = FilterEvent(ProjectionType.CLIP_EMBEDDING, set(range(0, 10)))
    with pytest.raises(ValueError, match="Invalid projection type: clip_embedding"):
        session.apply_filter_event(filter_event)
