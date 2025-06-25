"""Tests for quantization module."""

import pytest
import pandas as pd
from datetime import datetime

from crossfilter.core.quantization import (
    add_quantized_columns,
    get_optimal_h3_level,
    get_optimal_temporal_level, 
    aggregate_by_h3,
    aggregate_by_temporal,
    H3_LEVELS,
    TEMPORAL_LEVELS
)
from crossfilter.core.schema_constants import (
    SchemaColumns,
    TemporalLevel,
    DF_ID_COLUMN,
    get_h3_column_name,
    get_temporal_column_name
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: [f"uuid_{i}" for i in range(10)],
        SchemaColumns.GPS_LATITUDE: [37.7749 + i * 0.01 for i in range(10)],
        SchemaColumns.GPS_LONGITUDE: [-122.4194 + i * 0.01 for i in range(10)],
        SchemaColumns.TIMESTAMP_UTC: [
            f"2024-01-01T{10 + i}:00:00Z" for i in range(10)
        ]
    })
    # Convert timestamp to datetime
    df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(df[SchemaColumns.TIMESTAMP_UTC], utc=True)
    # Set stable df_id index
    df.index.name = DF_ID_COLUMN
    return df


def test_add_quantized_columns_spatial(sample_df, snapshot) -> None:
    """Test adding spatial quantization columns."""
    result = add_quantized_columns(sample_df)
    
    # Should have original columns plus quantized H3 columns
    assert len(result.columns) > len(sample_df.columns)
    
    # Check that H3 columns were added and their contents
    h3_columns = {col: result[col].tolist() for col in result.columns 
                  if col.startswith('QUANTIZED_H3_L')}
    assert h3_columns == snapshot


def test_add_quantized_columns_temporal(sample_df, snapshot) -> None:
    """Test adding temporal quantization columns."""
    result = add_quantized_columns(sample_df)
    
    # Check that temporal columns were added and their contents
    temporal_columns = {
        get_temporal_column_name(TemporalLevel.SECOND): result[get_temporal_column_name(TemporalLevel.SECOND)].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        get_temporal_column_name(TemporalLevel.MINUTE): result[get_temporal_column_name(TemporalLevel.MINUTE)].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        get_temporal_column_name(TemporalLevel.HOUR): result[get_temporal_column_name(TemporalLevel.HOUR)].dt.strftime('%Y-%m-%d %H:00').tolist(),
        get_temporal_column_name(TemporalLevel.DAY): result[get_temporal_column_name(TemporalLevel.DAY)].dt.strftime('%Y-%m-%d').tolist(),
        get_temporal_column_name(TemporalLevel.MONTH): result[get_temporal_column_name(TemporalLevel.MONTH)].dt.strftime('%Y-%m').tolist(),
        get_temporal_column_name(TemporalLevel.YEAR): result[get_temporal_column_name(TemporalLevel.YEAR)].dt.strftime('%Y').tolist()
    }
    assert temporal_columns == snapshot


def test_add_quantized_columns_missing_spatial(snapshot) -> None:
    """Test quantization when spatial columns are missing."""
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["uuid_1"],
        SchemaColumns.TIMESTAMP_UTC: [pd.Timestamp("2024-01-01T10:00:00Z", tz='UTC')]
    })
    df.index.name = DF_ID_COLUMN
    
    result = add_quantized_columns(df)
    
    # Check column presence and content
    column_info = {
        'present_columns': sorted(result.columns.tolist()),
        'temporal_hour_value': result[get_temporal_column_name(TemporalLevel.HOUR)].dt.strftime('%Y-%m-%d %H:00').iloc[0],
        'h3_columns_present': [col for col in result.columns if col.startswith('QUANTIZED_H3_L')]
    }
    assert column_info == snapshot


def test_add_quantized_columns_missing_temporal(snapshot) -> None:
    """Test quantization when temporal columns are missing."""
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["uuid_1"],
        SchemaColumns.GPS_LATITUDE: [37.7749],
        SchemaColumns.GPS_LONGITUDE: [-122.4194]
    })
    df.index.name = DF_ID_COLUMN
    
    result = add_quantized_columns(df)
    
    # Check column presence and content
    column_info = {
        'present_columns': sorted(result.columns.tolist()),
        'h3_columns_present': [col for col in result.columns if col.startswith('QUANTIZED_H3_L')],
        'temporal_columns_present': [col for col in result.columns if col.startswith('QUANTIZED_TIMESTAMP')],
        'h3_level_4_value': result[get_h3_column_name(H3_LEVELS[0])].iloc[0]
    }
    assert column_info == snapshot


def test_get_optimal_h3_level(sample_df) -> None:
    """Test finding optimal H3 level."""
    quantized = add_quantized_columns(sample_df)
    
    # With small dataset, should find a high-resolution level
    optimal = get_optimal_h3_level(quantized, max_groups=1000)
    assert optimal is not None
    assert optimal in H3_LEVELS
    
    # With very small max_groups, should find lower resolution
    optimal_small = get_optimal_h3_level(quantized, max_groups=1)
    assert optimal_small is not None
    assert optimal_small <= optimal  # Lower resolution has smaller numbers


def test_get_optimal_temporal_level(sample_df) -> None:
    """Test finding optimal temporal level."""
    quantized = add_quantized_columns(sample_df)
    
    # With small dataset, should find a fine-grained level
    optimal = get_optimal_temporal_level(quantized, max_groups=1000)
    assert optimal is not None
    assert optimal in TEMPORAL_LEVELS
    
    # With very small max_groups, should find coarser level
    optimal_small = get_optimal_temporal_level(quantized, max_groups=1)
    assert optimal_small is not None


def test_aggregate_by_h3(sample_df) -> None:
    """Test H3 aggregation."""
    quantized = add_quantized_columns(sample_df)
    h3_level = 7  # Use a specific level for testing
    
    aggregated = aggregate_by_h3(quantized, h3_level)
    
    # Should have aggregated columns
    expected_cols = [
        get_h3_column_name(h3_level),
        'lat', 'count', 'lon', 'df_ids'
    ]
    for col in expected_cols:
        assert col in aggregated.columns
    
    # Count should be sum of original rows
    assert aggregated['count'].sum() == len(sample_df)
    
    # df_ids should contain lists of original df_ids
    assert all(isinstance(df_ids, list) for df_ids in aggregated['df_ids'])
    
    # All original df_ids should be present
    all_df_ids = []
    for df_ids in aggregated['df_ids']:
        all_df_ids.extend(df_ids)
    assert set(all_df_ids) == set(sample_df.index)


def test_aggregate_by_temporal(sample_df) -> None:
    """Test temporal aggregation."""
    quantized = add_quantized_columns(sample_df)
    temporal_level = TemporalLevel.HOUR
    
    aggregated = aggregate_by_temporal(quantized, temporal_level)
    
    # Should have aggregated columns
    expected_cols = [
        f"QUANTIZED_TIMESTAMP_{temporal_level.upper()}",
        'count', 'df_ids', 'cumulative_count'
    ]
    for col in expected_cols:
        assert col in aggregated.columns
    
    # Count should be sum of original rows
    assert aggregated['count'].sum() == len(sample_df)
    
    # Cumulative count should end at total count
    assert aggregated['cumulative_count'].iloc[-1] == len(sample_df)
    
    # Should be sorted by timestamp
    timestamp_col = f"QUANTIZED_TIMESTAMP_{temporal_level.upper()}"
    assert aggregated[timestamp_col].is_monotonic_increasing


def test_aggregate_by_h3_invalid_level(sample_df) -> None:
    """Test H3 aggregation with invalid level."""
    quantized = add_quantized_columns(sample_df)
    
    with pytest.raises(ValueError, match="H3 level must be between 0 and 15, got 99"):
        aggregate_by_h3(quantized, 99)


def test_aggregate_by_temporal_invalid_level(sample_df) -> None:
    """Test temporal aggregation with invalid level."""
    quantized = add_quantized_columns(sample_df)
    
    with pytest.raises(ValueError, match="Temporal level invalid not found"):
        aggregate_by_temporal(quantized, "invalid")


def test_empty_dataframe() -> None:
    """Test quantization with empty DataFrame."""
    df = pd.DataFrame(columns=[
        SchemaColumns.GPS_LATITUDE,
        SchemaColumns.GPS_LONGITUDE,
        SchemaColumns.TIMESTAMP_UTC
    ])
    df.index.name = DF_ID_COLUMN
    
    result = add_quantized_columns(df)
    
    # Should not raise errors
    assert len(result) == 0
    
    # Should have quantized columns
    assert get_h3_column_name(H3_LEVELS[0]) in result.columns


def test_null_coordinates() -> None:
    """Test quantization with null coordinates."""
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["uuid_1", "uuid_2"],
        SchemaColumns.GPS_LATITUDE: [37.7749, None],
        SchemaColumns.GPS_LONGITUDE: [-122.4194, None],
        SchemaColumns.TIMESTAMP_UTC: [
            pd.Timestamp("2024-01-01T10:00:00Z", tz='UTC'),
            pd.Timestamp("2024-01-01T11:00:00Z", tz='UTC')
        ]
    })
    df.index.name = DF_ID_COLUMN
    
    result = add_quantized_columns(df)
    
    # H3 columns should have null for null coordinates
    h3_col = get_h3_column_name(H3_LEVELS[0])
    assert pd.notna(result.loc[0, h3_col])  # Valid coordinates
    assert pd.isna(result.loc[1, h3_col])   # Null coordinates
    
    # Temporal columns should work for both rows
    assert pd.notna(result.loc[0, get_temporal_column_name(TemporalLevel.HOUR)])
    assert pd.notna(result.loc[1, get_temporal_column_name(TemporalLevel.HOUR)])