"""Tests for bucketing module."""

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.bucketing import (
    H3_LEVELS,
    add_bucketed_columns,
    add_geo_h3_bucket_columns,
    add_temporal_bucket_columns,
    add_temporal_bucketed_columns,
    bucket_by_target_column,
    filter_df_to_selected_buckets,
    get_optimal_h3_level,
    get_optimal_temporal_level,
)
from crossfilter.core.schema import (
    DataType,
    TemporalLevel,
    get_h3_column_name,
    get_temporal_column_name,
)
from crossfilter.core.schema import (
    SchemaColumns as C,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
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


@pytest.fixture
def simple_data_example() -> pd.DataFrame:
    """Create simple test data with multiple rows per bucket to show filtering clearly."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1", "uuid_2", "uuid_3", "uuid_4", "uuid_5"],
            C.DATA_TYPE: [
                DataType.GPX_WAYPOINT,
                DataType.GPX_WAYPOINT,
                DataType.GPX_WAYPOINT,
                DataType.GPX_TRACKPOINT,
                DataType.GPX_TRACKPOINT,
            ],
            C.GPS_LATITUDE: [37.77, 37.78, 37.77, 37.79, 37.78],
            C.GPS_LONGITUDE: [-122.42, -122.43, -122.42, -122.44, -122.43],
            "category": ["A", "B", "A", "C", "B"],
            "value": [10, 20, 30, 40, 50],
        }
    )
    df.index.name = C.DF_ID
    return df


def test_thad_bucket_with_groupby(
    simple_data_example: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing with a groupby column."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=C.DATA_TYPE
    )
    assert bucketed.to_dict(orient="records") == snapshot


def test_add_quantized_columns_for_h3(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test adding H3 spatial quantization columns."""
    result = sample_df.copy()
    add_geo_h3_bucket_columns(result)
    assert result.dtypes.to_dict() == snapshot
    assert result.to_dict(orient="records") == snapshot


def test_add_quantized_columns_for_timestamp(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test adding temporal quantization columns."""
    result = sample_df.copy()
    add_temporal_bucket_columns(result)
    assert result.dtypes.to_dict() == snapshot
    assert result.to_dict(orient="records") == snapshot


def test_add_quantized_temporal_columns_no_timezone_warnings() -> None:
    """Test that temporal quantization does not emit timezone warnings."""
    import warnings

    # Create timezone-aware test data
    df = pd.DataFrame(
        {C.TIMESTAMP_UTC: pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")}
    )

    # Configure warnings to raise as exceptions
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Converting to PeriodArray/Index representation will drop timezone information.",
        )

        # This should not raise any warnings
        result = df.copy()
        add_temporal_bucket_columns(result)

        # Verify the result has the expected timezone-aware columns
        assert result[get_temporal_column_name(TemporalLevel.MONTH)].dt.tz is not None
        assert result[get_temporal_column_name(TemporalLevel.YEAR)].dt.tz is not None

        # Verify the timezone is preserved correctly
        assert str(result[get_temporal_column_name(TemporalLevel.MONTH)].dt.tz) == "UTC"
        assert str(result[get_temporal_column_name(TemporalLevel.YEAR)].dt.tz) == "UTC"


def test_add_quantized_columns_for_h3_missing_columns() -> None:
    """Test H3 quantization when spatial columns are missing."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1"],
            C.TIMESTAMP_UTC: [pd.Timestamp("2024-01-01T10:00:00Z", tz="UTC")],
        }
    )
    df.index.name = C.DF_ID

    with pytest.raises(KeyError):
        add_geo_h3_bucket_columns(df)


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

    with pytest.raises(KeyError):
        add_temporal_bucket_columns(df)


def test_get_optimal_h3_level(sample_df: pd.DataFrame) -> None:
    """Test finding optimal H3 level."""
    quantized = sample_df.copy()
    add_geo_h3_bucket_columns(quantized)

    # With large max_rows, should return None (no aggregation needed)
    optimal = get_optimal_h3_level(quantized, max_rows=1000)
    assert optimal is None

    # With very small max_rows, should find an H3 level for aggregation
    optimal_small = get_optimal_h3_level(quantized, max_rows=1)
    assert optimal_small is not None
    assert optimal_small in H3_LEVELS


def test_get_optimal_h3_level_large_dataset_issue() -> None:
    """Test H3 level selection with large dataset to reproduce the issue."""
    # Create a large dataset spread across different locations to simulate ~100k rows
    import numpy as np

    # Create diverse coordinates that span different H3 cells
    np.random.seed(42)  # For reproducible results
    size = 10000  # Smaller for testing but still shows the issue

    # Generate coordinates spread across a large geographic area
    # This ensures we get diverse H3 cells at different levels
    lats = np.random.uniform(37.0, 38.0, size)  # SF Bay Area roughly
    lons = np.random.uniform(-123.0, -122.0, size)

    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(size)],
            C.GPS_LATITUDE: lats,
            C.GPS_LONGITUDE: lons,
        }
    )
    df.index.name = C.DF_ID

    # Add quantized H3 columns
    quantized = df.copy()
    add_geo_h3_bucket_columns(quantized)

    # Target 1000 rows (10% of our data)
    target_rows = 1000
    optimal_level = get_optimal_h3_level(quantized, max_rows=target_rows)

    # Check that we got a reasonable level (not 0)
    assert optimal_level is not None

    # Get the actual bucket count for this level
    h3_col = get_h3_column_name(optimal_level)
    actual_buckets = quantized[h3_col].nunique()

    # The optimal level should produce close to target_rows buckets
    # It should be <= target_rows but not dramatically smaller
    assert actual_buckets <= target_rows

    # Verify that the algorithm chose a reasonable level that produces a good number of buckets
    # (not too few, which would indicate it chose too coarse a resolution)
    assert (
        actual_buckets > target_rows / 10
    ), f"Got {actual_buckets} buckets, expected more than {target_rows/10}"


def test_get_optimal_temporal_level(snapshot: SnapshotAssertion) -> None:
    """Test finding optimal temporal level."""
    length = 10000
    df = pd.DataFrame(
        {
            C.UUID_STRING: [f"uuid_{i}" for i in range(length)],
            C.TIMESTAMP_UTC: [
                pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(seconds=i)
                for i in range(10000)
            ],
        }
    )
    quantized = df.copy()
    add_temporal_bucket_columns(quantized)
    actual = {
        probe: get_optimal_temporal_level(quantized, max_rows=probe)
        for probe in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 5000, 10000]
    }
    assert actual == snapshot


def test_bucket_by_target_column_h3(sample_df: pd.DataFrame) -> None:
    """Test bucketing by H3 column."""
    # First add H3 quantization columns
    quantized = sample_df.copy()
    add_geo_h3_bucket_columns(quantized)
    h3_level = 7  # Use a specific level for testing
    target_column = get_h3_column_name(h3_level)

    bucketed = bucket_by_target_column(quantized, target_column, groupby_column=None)

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
    quantized = sample_df.copy()
    add_temporal_bucket_columns(quantized)
    temporal_level = TemporalLevel.HOUR
    target_column = get_temporal_column_name(temporal_level)

    bucketed = bucket_by_target_column(quantized, target_column, groupby_column=None)

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
    with pytest.raises(
        ValueError, match="Target column 'invalid_column' not found in DataFrame"
    ):
        bucket_by_target_column(sample_df, "invalid_column", groupby_column=None)


def test_add_quantized_columns_for_h3_empty_dataframe() -> None:
    """Test H3 quantization with empty DataFrame."""
    df = pd.DataFrame(
        columns=[
            C.GPS_LATITUDE,
            C.GPS_LONGITUDE,
        ]
    )
    df.index.name = C.DF_ID

    result = df.copy()
    add_geo_h3_bucket_columns(result)

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

    result = df.copy()
    add_temporal_bucket_columns(result)

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

    result = df.copy()
    add_geo_h3_bucket_columns(result)

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

    bucketed = bucket_by_target_column(df, "test_column", groupby_column=None)

    # Should handle nulls gracefully
    assert len(bucketed) == 1  # One for "A", null is dropped.
    assert bucketed["COUNT"].sum() == 2  # All original rows counted


def test_bucket_by_target_column_h3_column_preservation_snapshot(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing by H3 column with snapshot to show column preservation."""
    # First add H3 quantization columns
    quantized = sample_df.copy()
    add_geo_h3_bucket_columns(quantized)
    h3_level = 7  # Use a specific level for testing
    target_column = get_h3_column_name(h3_level)

    bucketed = bucket_by_target_column(quantized, target_column, groupby_column=None)

    # Snapshot the bucketed result to show column preservation
    assert bucketed.to_dict(orient="records") == snapshot


def test_bucket_by_target_column_temporal_column_preservation_snapshot(
    sample_df: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing by temporal column with snapshot to show column preservation."""
    # First add temporal quantization columns
    quantized = sample_df.copy()
    add_temporal_bucket_columns(quantized)
    temporal_level = TemporalLevel.HOUR
    target_column = get_temporal_column_name(temporal_level)

    bucketed = bucket_by_target_column(quantized, target_column, groupby_column=None)

    # Snapshot the bucketed result to show column preservation
    assert bucketed.to_dict(orient="records") == snapshot


def test_bucket_by_target_column_simple_data_snapshot(
    simple_data_example: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test bucketing with simple test data to clearly show column preservation."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Snapshot shows:
    # - Each unique category becomes one row
    # - Other columns get first values from each bucket
    # - COUNT shows how many original rows per bucket
    assert bucketed.to_dict(orient="records") == snapshot


def test_filter_df_to_selected_buckets_basic(simple_data_example: pd.DataFrame) -> None:
    """Test basic filtering of original data to selected buckets."""
    # Create bucketed data
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Select bucket index 0 (category "A") and 2 (category "C")
    # Based on the data: A appears in rows 0,2 and C appears in row 3
    selected_indices = [0, 2]

    filtered = filter_df_to_selected_buckets(
        simple_data_example, bucketed, "category", selected_indices
    )

    # Should get rows where category is "A" or "C"
    expected_categories = {"A", "C"}
    assert set(filtered["category"]) == expected_categories
    assert len(filtered) == 3  # 2 rows for "A", 1 row for "C"

    # Check specific UUIDs are included
    expected_uuids = {"uuid_1", "uuid_3", "uuid_4"}  # Rows 0, 2, 3 from original
    assert set(filtered[C.UUID_STRING]) == expected_uuids


def test_filter_df_to_selected_buckets_single_bucket(
    simple_data_example: pd.DataFrame,
) -> None:
    """Test filtering to a single bucket."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Select only bucket index 1 (category "B")
    selected_indices = [1]

    filtered = filter_df_to_selected_buckets(
        simple_data_example, bucketed, "category", selected_indices
    )

    # Should get only rows where category is "B"
    assert len(filtered) == 2  # 2 rows for "B"
    assert all(filtered["category"] == "B")
    expected_uuids = {"uuid_2", "uuid_5"}  # Rows 1, 4 from original
    assert set(filtered[C.UUID_STRING]) == expected_uuids


def test_filter_df_to_selected_buckets_all_buckets(
    simple_data_example: pd.DataFrame,
) -> None:
    """Test filtering when all buckets are selected."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Select all bucket indices
    selected_indices = [0, 1, 2]  # All buckets

    filtered = filter_df_to_selected_buckets(
        simple_data_example, bucketed, "category", selected_indices
    )

    # Should get all original rows
    assert len(filtered) == len(simple_data_example)
    pd.testing.assert_frame_equal(
        filtered.sort_index(), simple_data_example.sort_index()
    )


def test_thad_filter_df_to_selected_buckets_with_groupby() -> None:
    """Test filtering when all buckets are selected."""
    df = pd.DataFrame(
        {
            C.UUID_STRING: ["uuid_1", "uuid_2", "uuid_3", "uuid_4", "uuid_5"],
            C.DATA_TYPE: [
                DataType.GPX_WAYPOINT,
                DataType.GPX_WAYPOINT,
                DataType.GPX_TRACKPOINT,
                DataType.GPX_TRACKPOINT,
                DataType.GPX_WAYPOINT,
            ],
            C.GPS_LATITUDE: [37.77, 37.78, 37.77, 37.79, 37.78],
            C.GPS_LONGITUDE: [-122.42, -122.43, -122.42, -122.44, -122.43],
            "category": ["A", "B", "A", "C", "B"],
            "value": [10, 20, 30, 40, 50],
        }
    )

    bucketed = bucket_by_target_column(df, "category", groupby_column=C.DATA_TYPE)

    selected_indices = bucketed.query(
        "(category == 'A' and DATA_TYPE == 'GPX_TRACKPOINT') or (category == 'B' and DATA_TYPE == 'GPX_WAYPOINT')"
    ).index.tolist()
    assert selected_indices == [1, 2]

    filtered = filter_df_to_selected_buckets(
        df, bucketed, "category", selected_indices, groupby_column=C.DATA_TYPE
    )

    # Should get all original rows
    assert filtered.index.tolist() == [1, 2, 4]


def test_filter_df_to_selected_buckets_empty_selection(
    simple_data_example: pd.DataFrame,
) -> None:
    """Test filtering with empty bucket selection."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Select no buckets
    selected_indices = []

    filtered = filter_df_to_selected_buckets(
        simple_data_example, bucketed, "category", selected_indices
    )

    # Should get empty DataFrame with same structure
    assert len(filtered) == 0
    assert list(filtered.columns) == list(simple_data_example.columns)
    assert filtered.index.name == simple_data_example.index.name


def test_filter_df_to_selected_buckets_invalid_indices(
    simple_data_example: pd.DataFrame,
) -> None:
    """Test error handling for invalid bucket indices."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Test negative index
    with pytest.raises(KeyError):
        filter_df_to_selected_buckets(simple_data_example, bucketed, "category", [-1])

    # Test index too large
    with pytest.raises(KeyError):
        filter_df_to_selected_buckets(simple_data_example, bucketed, "category", [5])

    # Test multiple invalid indices
    with pytest.raises(KeyError):
        filter_df_to_selected_buckets(
            simple_data_example, bucketed, "category", [-1, 1, 10]
        )


def test_filter_df_to_selected_buckets_missing_target_column(
    simple_data_example: pd.DataFrame,
) -> None:
    """Test error handling when target column is missing."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Test missing column in original data
    with pytest.raises(ValueError):
        filter_df_to_selected_buckets(simple_data_example, bucketed, "missing_col", [0])

    # Test missing column in bucketed data - create bucketed df without the target column
    bucketed_missing_col = bucketed.drop(columns=["category"])
    with pytest.raises(KeyError):
        filter_df_to_selected_buckets(
            simple_data_example, bucketed_missing_col, "category", [0]
        )


def test_filter_df_to_selected_buckets_snapshot(
    simple_data_example: pd.DataFrame, snapshot: SnapshotAssertion
) -> None:
    """Test filtering result with snapshot to show exact filtering behavior."""
    bucketed = bucket_by_target_column(
        simple_data_example, "category", groupby_column=None
    )

    # Select buckets 0 and 2 (categories "A" and "C")
    selected_indices = [0, 2]

    filtered = filter_df_to_selected_buckets(
        simple_data_example, bucketed, "category", selected_indices
    )

    # Snapshot the filtered result
    assert filtered.to_dict(orient="records") == snapshot


def test_add_bucketed_columns(snapshot: SnapshotAssertion) -> None:
    """Test the combined add_bucketed_columns function."""
    # Create test data with both spatial and temporal columns
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2", "uuid3"],
        C.GPS_LATITUDE: [37.7749, 37.7849, 37.7949],
        C.GPS_LONGITUDE: [-122.4194, -122.4094, -122.3994],
        C.TIMESTAMP_UTC: pd.to_datetime(
            [
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 12:00:00",
            ],
            utc=True,
        ),
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID
    result = add_bucketed_columns(df)
    assert result.to_dict(orient="records") == snapshot


def test_add_bucketed_columns_spatial_only(snapshot: SnapshotAssertion) -> None:
    """Test add_bucketed_columns with only spatial data."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        C.GPS_LATITUDE: [37.7749, 37.7849],
        C.GPS_LONGITUDE: [-122.4194, -122.4094],
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_bucketed_columns(df)
    assert result.to_dict(orient="records") == snapshot


def test_add_bucketed_columns_temporal_only(snapshot: SnapshotAssertion) -> None:
    """Test add_bucketed_columns with only temporal data."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        C.TIMESTAMP_UTC: pd.to_datetime(
            ["2024-01-01 10:00:00", "2024-01-01 11:00:00"], utc=True
        ),
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_bucketed_columns(df)
    assert result.to_dict(orient="records") == snapshot


def test_add_bucketed_columns_neither(snapshot: SnapshotAssertion) -> None:
    """Test add_bucketed_columns with neither spatial nor temporal data."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        "some_other_column": ["value1", "value2"],
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_bucketed_columns(df)
    assert result.to_dict(orient="records") == snapshot


def test_add_temporal_bucketed_columns(snapshot: SnapshotAssertion) -> None:
    """Test the new add_temporal_bucketed_columns function."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        C.TIMESTAMP_UTC: pd.to_datetime(
            ["2024-01-01 10:00:00", "2024-01-01 11:00:00"], utc=True
        ),
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_temporal_bucketed_columns(df)

    assert result.to_dict(orient="records") == snapshot


def test_add_temporal_bucketed_columns_with_spatial_data(
    snapshot: SnapshotAssertion,
) -> None:
    """Test add_temporal_bucketed_columns ignores spatial data."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        C.GPS_LATITUDE: [37.7749, 37.7849],
        C.GPS_LONGITUDE: [-122.4194, -122.4094],
        C.TIMESTAMP_UTC: pd.to_datetime(
            ["2024-01-01 10:00:00", "2024-01-01 11:00:00"], utc=True
        ),
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_temporal_bucketed_columns(df)
    assert result.to_dict(orient="records") == snapshot


def test_add_temporal_bucketed_columns_no_temporal_data(
    snapshot: SnapshotAssertion,
) -> None:
    """Test add_temporal_bucketed_columns with no temporal data."""
    test_data = {
        C.UUID_STRING: ["uuid1", "uuid2"],
        "some_other_column": ["value1", "value2"],
    }
    df = pd.DataFrame(test_data)
    df.index.name = C.DF_ID

    result = add_temporal_bucketed_columns(df)

    # Should have no additional columns
    temporal_cols = [
        col for col in result.columns if col.startswith("QUANTIZED_TIMESTAMP_")
    ]

    assert len(temporal_cols) == 0
    assert len(result.columns) == len(df.columns)
