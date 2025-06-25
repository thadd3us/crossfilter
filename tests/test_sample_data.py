"""Tests for sample test data loading."""

from pathlib import Path
import pytest
import pandas as pd

from crossfilter.core.data_schema import load_jsonl_to_dataframe, DataSchema
from crossfilter.core.schema_constants import SchemaColumns


def test_sample_100_jsonl_loads() -> None:
    """Test that sample_100.jsonl loads correctly and follows the schema."""
    sample_file = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    
    # Verify the file exists
    assert sample_file.exists(), f"Sample file not found: {sample_file}"
    
    # Load the data
    df = load_jsonl_to_dataframe(sample_file)
    
    # Verify we have 100 records
    assert len(df) == 100, f"Expected 100 records, got {len(df)}"
    
    # Verify required columns are present
    required_columns = [
        SchemaColumns.UUID_STRING, SchemaColumns.GPS_LATITUDE, SchemaColumns.GPS_LONGITUDE, SchemaColumns.TIMESTAMP_UTC
    ]
    for col in required_columns:
        assert col in df.columns, f"Required column {col} missing"
        assert not df[col].isna().all(), f"Column {col} is all null"
    
    # Verify UUID_STRING values are present and unique
    assert df[SchemaColumns.UUID_STRING].nunique() == 100, "UUID_STRING should have 100 unique values"
    
    # Verify GPS coordinates are within valid ranges
    assert df[SchemaColumns.GPS_LATITUDE].between(-90, 90).all(), "GPS_LATITUDE out of range"
    assert df[SchemaColumns.GPS_LONGITUDE].between(-180, 180).all(), "GPS_LONGITUDE out of range"
    
    # Verify TIMESTAMP_UTC is properly parsed as timezone-aware datetime
    assert df[SchemaColumns.TIMESTAMP_UTC].dtype.name == "datetime64[ns, UTC]", "TIMESTAMP_UTC should be UTC timezone-aware"
    
    # Verify schema validation passes
    try:
        DataSchema.validate(df)
    except Exception as e:
        pytest.fail(f"Schema validation failed: {e}")
    
    # Verify we have some variety in DATA_TYPE
    data_types = df[SchemaColumns.DATA_TYPE].dropna().unique()
    assert len(data_types) > 1, "Should have multiple DATA_TYPE values"
    
    # Verify all DATA_TYPE values are valid
    valid_data_types = ["PHOTO", "VIDEO", "GPX_TRACKPOINT", "GPX_WAYPOINT"]
    for dt in data_types:
        assert dt in valid_data_types, f"Invalid DATA_TYPE: {dt}"


def test_sample_data_has_geographic_diversity() -> None:
    """Test that sample data covers multiple geographic locations."""
    sample_file = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    df = load_jsonl_to_dataframe(sample_file)
    
    # Should have significant latitude range (multiple cities)
    lat_range = df[SchemaColumns.GPS_LATITUDE].max() - df[SchemaColumns.GPS_LATITUDE].min()
    assert lat_range > 50, f"Expected wide latitude range, got {lat_range}"
    
    # Should have significant longitude range (multiple continents)
    lon_range = df[SchemaColumns.GPS_LONGITUDE].max() - df[SchemaColumns.GPS_LONGITUDE].min()
    assert lon_range > 200, f"Expected wide longitude range, got {lon_range}"


def test_sample_data_has_temporal_diversity() -> None:
    """Test that sample data covers multiple time periods."""
    sample_file = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    df = load_jsonl_to_dataframe(sample_file)
    
    # Should span multiple days
    time_range = df[SchemaColumns.TIMESTAMP_UTC].max() - df[SchemaColumns.TIMESTAMP_UTC].min()
    assert time_range.days >= 9, f"Expected at least 9 days span, got {time_range.days}"