"""Tests for data schema module."""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path

from crossfilter.core.data_schema import DataSchema, DataType, load_jsonl_to_dataframe
from crossfilter.core.schema_constants import SchemaColumns, DF_ID_COLUMN


# THAD: Always specify return type annotations for all functions.
def test_data_type_enum():
    """Test DataType enum values."""
    assert DataType.PHOTO == "PHOTO"
    assert DataType.VIDEO == "VIDEO"
    assert DataType.GPX_TRACKPOINT == "GPX_TRACKPOINT"
    assert DataType.GPX_WAYPOINT == "GPX_WAYPOINT"


def test_load_jsonl_empty_file():
    """Test loading an empty JSONL file."""
    # THAD: use pytest's tmp_path fixture instead of tempfile.NamedTemporaryFile, and Path.write_text to fill it.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write("")
        temp_path = Path(f.name)
    
    try:
        df = load_jsonl_to_dataframe(temp_path)
        assert len(df) == 0
        assert df.index.name == DF_ID_COLUMN
        # Should have all schema columns
        for col in SchemaColumns:
            assert col in df.columns
    finally:
        temp_path.unlink()


def test_load_jsonl_sample_data():
    """Test loading sample JSONL data."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            "TIMESTAMP_UTC": "2024-01-01T10:00:00Z",
            "DATA_TYPE": "PHOTO",
            "NAME": "Test Photo"
        },
        {
            "UUID_STRING": "test-uuid-2", 
            "GPS_LATITUDE": 37.7849,
            "GPS_LONGITUDE": -122.4094,
            "TIMESTAMP_UTC": "2024-01-01T10:05:00Z",
            "DATA_TYPE": "GPX_TRACKPOINT"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        temp_path = Path(f.name)
    
    try:
        df = load_jsonl_to_dataframe(temp_path)
        # THAD: Lean on pytest's syrupy plugin to check the contents of data in a way that's easy to diff and keeps the test short.
        assert len(df) == 2
        assert df.index.name == DF_ID_COLUMN
        assert list(df.index) == [0, 1]  # df_id should be 0, 1
        
        # Check specific values
        assert df.loc[0, SchemaColumns.UUID_STRING] == "test-uuid-1"
        assert df.loc[0, SchemaColumns.GPS_LATITUDE] == 37.7749
        assert df.loc[1, SchemaColumns.DATA_TYPE] == "GPX_TRACKPOINT"
        
        # Check timestamp conversion
        assert pd.api.types.is_datetime64tz_dtype(df[SchemaColumns.TIMESTAMP_UTC])
        
    finally:
        temp_path.unlink()


def test_load_jsonl_missing_columns():
    """Test that missing schema columns are added with null values."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            # Missing most columns
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        temp_path = Path(f.name)
    
    try:
        df = load_jsonl_to_dataframe(temp_path)
        assert len(df) == 1
        
        # Check that all schema columns exist
        for col in SchemaColumns:
            assert col in df.columns
            
        # Missing columns should be null except the ones we provided
        assert df.loc[0, SchemaColumns.UUID_STRING] == "test-uuid-1"
        assert pd.isna(df.loc[0, SchemaColumns.NAME])
        assert pd.isna(df.loc[0, SchemaColumns.TIMESTAMP_UTC])
        
    finally:
        temp_path.unlink()


def test_load_jsonl_extra_columns():
    """Test that extra columns are preserved."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            "EXTRA_COLUMN": "extra_value",
            "ANOTHER_EXTRA": 42
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        temp_path = Path(f.name)
    
    try:
        df = load_jsonl_to_dataframe(temp_path)
        assert len(df) == 1
        
        # Extra columns should be preserved
        assert "EXTRA_COLUMN" in df.columns
        assert "ANOTHER_EXTRA" in df.columns
        assert df.loc[0, "EXTRA_COLUMN"] == "extra_value"
        assert df.loc[0, "ANOTHER_EXTRA"] == 42
        
    finally:
        temp_path.unlink()


def test_load_jsonl_nonexistent_file():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_jsonl_to_dataframe(Path("/nonexistent/file.jsonl"))


def test_load_jsonl_invalid_json():
    """Test that invalid JSON raises ValueError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"valid": "json"}\n')
        f.write('invalid json line\n')
        f.write('{"another": "valid"}\n')
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_jsonl_to_dataframe(temp_path)
    finally:
        temp_path.unlink()


def test_df_id_stability():
    """Test that df_id remains stable across operations."""
    sample_data = [
        {"UUID_STRING": "uuid-1", "GPS_LATITUDE": 37.7749},
        {"UUID_STRING": "uuid-2", "GPS_LATITUDE": 37.7849},
        {"UUID_STRING": "uuid-3", "GPS_LATITUDE": 37.7949}
    ]
    
    # THAD: use pytest's tmp_path fixture instead of tempfile.NamedTemporaryFile, and Path.write_text to fill it.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        temp_path = Path(f.name)
    
    try:
        df = load_jsonl_to_dataframe(temp_path)
        
        # df_id should be stable integer index
        assert list(df.index) == [0, 1, 2]
        assert df.index.name == DF_ID_COLUMN
        
        # Test that we can reference rows by df_id
        assert df.loc[1, SchemaColumns.UUID_STRING] == "uuid-2"
        
        # Test filtering by df_id
        filtered = df.loc[[0, 2]]
        assert len(filtered) == 2
        assert list(filtered.index) == [0, 2]
        
    finally:
        temp_path.unlink()