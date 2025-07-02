"""Tests for bucketing module."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.bucketing import (
    H3_LEVELS,
    TEMPORAL_LEVELS,
    add_quantized_columns_for_h3,
    add_quantized_columns_for_timestamp,
    bucket_by_target_column,
    get_optimal_h3_level,
    get_optimal_temporal_level,
)
from crossfilter.core.schema import (
    SchemaColumns as C,
    TemporalLevel,
    get_h3_column_name,
    get_temporal_column_name,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(10)],
            C.GPS_LATITUDE: [37.7749 + i * 0.01 for i in range(10)],
            C.GPS_LONGITUDE: [-122.4194 + i * 0.01 for i in range(10)],
            C.TIMESTAMP_UTC: [f"2024-01-01T{10 + i}:00:00Z" for i in range(10)],
        }
    )
    # Convert timestamp to datetime
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    # Set stable df_id index
    df.index.name = C.DF_ID
    return df


def test_add_quantized_columns_for_h3(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test adding H3 spatial quantization columns."""
    result = add_quantized_columns_for_h3(sample_df)
    assert result.dtypes.to_dict() == snapshot
    assert result.to_dict(orient="records") == snapshot


def test_add_quantized_columns_for_timestamp(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test adding temporal quantization columns."""
    result = add_quantized_columns_for_timestamp(sample_df)
    assert result.dtypes.to_dict() == snapshot
    assert result.to_dict(orient="records") == snapshot


def test_add_quantized_columns_for_h3_missing_columns() -> None:
    """Test H3 quantization when spatial columns are missing."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1"],
            C.TIMESTAMP_UTC: [pd.Timestamp("2024-01-01T10:00:00Z", tz="UTC")],
        }
    )
    df.index.name = C.DF_ID
    
    with pytest.raises(ValueError, match="DataFrame must contain GPS_LATITUDE and GPS_LONGITUDE columns"):
        add_quantized_columns_for_h3(df)


def test_add_quantized_columns_for_timestamp_missing_column() -> None:
    """Test temporal quantization when temporal column is missing."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1"],
            C.GPS_LATITUDE: [37.7749],
            C.GPS_LONGITUDE: [-122.4194],
        }
    )
    df.index.name = C.DF_ID

    with pytest.raises(ValueError, match="DataFrame must contain TIMESTAMP_UTC column"):
        add_quantized_columns_for_timestamp(df)


def test_get_optimal_h3_level(sample_df: pd.DataFrame) -> None:
    """Test finding optimal H3 level."""
    quantized = add_quantized_columns_for_h3(sample_df)

    # With small dataset, should find a high-resolution level
    optimal = get_optimal_h3_level(quantized, max_groups=1000)
    assert optimal is not None
    assert optimal in H3_LEVELS

    # With very small max_groups, should find lower resolution
    optimal_small = get_optimal_h3_level(quantized, max_groups=1)
    assert optimal_small is not None
    assert optimal_small <= optimal  # Lower resolution has smaller numbers


def test_get_optimal_temporal_level(sample_df: pd.DataFrame) -> None:
    """Test finding optimal temporal level."""
    quantized = add_quantized_columns_for_timestamp(sample_df)

    # With small dataset, should find a fine-grained level
    optimal = get_optimal_temporal_level(quantized, max_groups=1000)
    assert optimal is not None
    assert optimal in TEMPORAL_LEVELS

    # With very small max_groups, should find coarser level
    optimal_small = get_optimal_temporal_level(quantized, max_groups=1)
    assert optimal_small is not None


def test_bucket_by_target_column_h3(sample_df: pd.DataFrame) -> None:
    """Test bucketing by H3 column."""
    # First add H3 quantization columns
    quantized = add_quantized_columns_for_h3(sample_df)
    h3_level = 7  # Use a specific level for testing
    target_column = get_h3_column_name(h3_level)

    bucketed = bucket_by_target_column(quantized, target_column)

    # Should have same column structure plus COUNT
    expected_cols = set(quantized.columns) | {"COUNT"}
    assert set(bucketed.columns) == expected_cols

    # COUNT should sum to original row count
    assert bucketed["COUNT"].sum() == len(sample_df)

    # Should have standard integer index
    assert bucketed.index.name == C.DF_ID
    assert isinstance(bucketed.index, pd.RangeIndex)

    # Should have one row per unique H3 cell
    assert len(bucketed) == quantized[target_column].nunique()


def test_bucket_by_target_column_temporal(sample_df: pd.DataFrame) -> None:
    """Test bucketing by temporal column."""
    # First add temporal quantization columns
    quantized = add_quantized_columns_for_timestamp(sample_df)
    temporal_level = TemporalLevel.HOUR
    target_column = get_temporal_column_name(temporal_level)

    bucketed = bucket_by_target_column(quantized, target_column)

    # Should have same column structure plus COUNT
    expected_cols = set(quantized.columns) | {"COUNT"}
    assert set(bucketed.columns) == expected_cols

    # COUNT should sum to original row count
    assert bucketed["COUNT"].sum() == len(sample_df)

    # Should have standard integer index
    assert bucketed.index.name == C.DF_ID
    assert isinstance(bucketed.index, pd.RangeIndex)

    # Should have one row per unique temporal bucket
    assert len(bucketed) == quantized[target_column].nunique()


def test_bucket_by_target_column_invalid_column(sample_df: pd.DataFrame) -> None:
    """Test bucketing with invalid target column."""
    with pytest.raises(ValueError, match="Target column 'invalid_column' not found in DataFrame"):
        bucket_by_target_column(sample_df, "invalid_column")


def test_add_quantized_columns_for_h3_empty_dataframe() -> None:
    """Test H3 quantization with empty DataFrame."""
    df = pd.DataFrame(
        columns=[
            C.GPS_LATITUDE,
            C.GPS_LONGITUDE,
        ]
    )
    df.index.name = C.DF_ID

    result = add_quantized_columns_for_h3(df)

    # Should not raise errors
    assert len(result) == 0

    # Should have H3 quantized columns
    assert get_h3_column_name(H3_LEVELS[0]) in result.columns


def test_add_quantized_columns_for_timestamp_empty_dataframe() -> None:
    """Test temporal quantization with empty DataFrame."""
    df = pd.DataFrame(
        columns=[
            C.TIMESTAMP_UTC,
        ]
    )
    df.index.name = C.DF_ID

    result = add_quantized_columns_for_timestamp(df)

    # Should not raise errors
    assert len(result) == 0

    # Should have temporal quantized columns
    assert get_temporal_column_name(TemporalLevel.HOUR) in result.columns


def test_add_quantized_columns_for_h3_null_coordinates() -> None:
    """Test H3 quantization with null coordinates."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1", "uuid_2"],
            C.GPS_LATITUDE: [37.7749, None],
            C.GPS_LONGITUDE: [-122.4194, None],
        }
    )
    df.index.name = C.DF_ID

    result = add_quantized_columns_for_h3(df)

    # H3 columns should have null for null coordinates
    h3_col = get_h3_column_name(H3_LEVELS[0])
    assert pd.notna(result.loc[0, h3_col])  # Valid coordinates
    assert pd.isna(result.loc[1, h3_col])  # Null coordinates


def test_bucket_by_target_column_with_nulls() -> None:
    """Test bucketing with null values in target column."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1", "uuid_2", "uuid_3"],
            C.GPS_LATITUDE: [37.7749, 37.7750, 37.7749],
            C.GPS_LONGITUDE: [-122.4194, -122.4195, -122.4194],
            "test_column": ["A", None, "A"],
        }
    )
    df.index.name = C.DF_ID

    bucketed = bucket_by_target_column(df, "test_column")

    # Should handle nulls gracefully
    assert len(bucketed) == 2  # One for "A", one for null
    assert bucketed["COUNT"].sum() == 3  # All original rows counted


def test_bucket_by_target_column_h3_column_preservation_snapshot(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing by H3 column with snapshot to show column preservation."""
    # First add H3 quantization columns
    quantized = add_quantized_columns_for_h3(sample_df)
    h3_level = 7  # Use a specific level for testing
    target_column = get_h3_column_name(h3_level)

    bucketed = bucket_by_target_column(quantized, target_column)
    
    # Snapshot the bucketed result to show column preservation
    assert bucketed.to_dict(orient="records") == snapshot


def test_bucket_by_target_column_temporal_column_preservation_snapshot(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing by temporal column with snapshot to show column preservation."""
    # First add temporal quantization columns
    quantized = add_quantized_columns_for_timestamp(sample_df)
    temporal_level = TemporalLevel.HOUR
    target_column = get_temporal_column_name(temporal_level)

    bucketed = bucket_by_target_column(quantized, target_column)
    
    # Snapshot the bucketed result to show column preservation
    assert bucketed.to_dict(orient="records") == snapshot


def test_bucket_by_target_column_simple_data_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test bucketing with simple test data to clearly show column preservation."""
    # Create simple test data with multiple rows per bucket to show preservation
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1", "uuid_2", "uuid_3", "uuid_4", "uuid_5"],
            C.GPS_LATITUDE: [37.77, 37.78, 37.77, 37.79, 37.78],
            C.GPS_LONGITUDE: [-122.42, -122.43, -122.42, -122.44, -122.43],
            "category": ["A", "B", "A", "C", "B"],
            "value": [10, 20, 30, 40, 50],
        }
    )
    df.index.name = C.DF_ID

    bucketed = bucket_by_target_column(df, "category")
    
    # Snapshot shows:
    # - Each unique category becomes one row
    # - Other columns get first values from each bucket
    # - COUNT shows how many original rows per bucket
    assert bucketed.to_dict(orient="records") == snapshot
