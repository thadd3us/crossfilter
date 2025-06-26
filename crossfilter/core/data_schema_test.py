"""Tests for data schema module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from crossfilter.core.data_schema import DataType, load_jsonl_to_dataframe
from crossfilter.core.schema_constants import DF_ID_COLUMN, SchemaColumns


def test_data_type_enum() -> None:
    """Test DataType enum values."""
    assert DataType.PHOTO == "PHOTO"
    assert DataType.VIDEO == "VIDEO"
    assert DataType.GPX_TRACKPOINT == "GPX_TRACKPOINT"
    assert DataType.GPX_WAYPOINT == "GPX_WAYPOINT"


def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    """Test loading an empty JSONL file."""
    temp_file = tmp_path / "empty.jsonl"
    temp_file.write_text("")

    df = load_jsonl_to_dataframe(temp_file)
    assert len(df) == 0
    assert df.index.name == DF_ID_COLUMN
    # Should have all schema columns
    for col in SchemaColumns:
        assert col in df.columns


def test_load_jsonl_sample_data(tmp_path: Path) -> None:
    """Test loading sample JSONL data."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            "TIMESTAMP_UTC": "2024-01-01T10:00:00Z",
            "DATA_TYPE": "PHOTO",
            "NAME": "Test Photo",
        },
        {
            "UUID_STRING": "test-uuid-2",
            "GPS_LATITUDE": 37.7849,
            "GPS_LONGITUDE": -122.4094,
            "TIMESTAMP_UTC": "2024-01-01T10:05:00Z",
            "DATA_TYPE": "GPX_TRACKPOINT",
        },
    ]

    temp_file = tmp_path / "sample.jsonl"
    temp_file.write_text("\n".join(json.dumps(record) for record in sample_data) + "\n")

    df = load_jsonl_to_dataframe(temp_file)

    # Basic structure checks
    assert len(df) == 2
    assert df.index.name == DF_ID_COLUMN
    assert list(df.index) == [0, 1]  # df_id should be 0, 1

    # Check specific values
    assert df.loc[0, SchemaColumns.UUID_STRING] == "test-uuid-1"
    assert df.loc[0, SchemaColumns.GPS_LATITUDE] == 37.7749
    assert df.loc[1, SchemaColumns.DATA_TYPE] == "GPX_TRACKPOINT"

    # Check timestamp conversion
    assert pd.api.types.is_datetime64tz_dtype(df[SchemaColumns.TIMESTAMP_UTC])


def test_load_jsonl_missing_columns(tmp_path: Path) -> None:
    """Test that missing schema columns are added with null values."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            # Missing most columns
        }
    ]

    temp_file = tmp_path / "missing_cols.jsonl"
    temp_file.write_text("\n".join(json.dumps(record) for record in sample_data) + "\n")

    df = load_jsonl_to_dataframe(temp_file)
    assert len(df) == 1

    # Check that all schema columns exist
    for col in SchemaColumns:
        assert col in df.columns

    # Missing columns should be null except the ones we provided
    assert df.loc[0, SchemaColumns.UUID_STRING] == "test-uuid-1"
    assert pd.isna(df.loc[0, SchemaColumns.NAME])
    assert pd.isna(df.loc[0, SchemaColumns.TIMESTAMP_UTC])


def test_load_jsonl_extra_columns(tmp_path: Path) -> None:
    """Test that extra columns are preserved."""
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
            "EXTRA_COLUMN": "extra_value",
            "ANOTHER_EXTRA": 42,
        }
    ]

    temp_file = tmp_path / "extra_cols.jsonl"
    temp_file.write_text("\n".join(json.dumps(record) for record in sample_data) + "\n")

    df = load_jsonl_to_dataframe(temp_file)
    assert len(df) == 1

    # Extra columns should be preserved
    assert "EXTRA_COLUMN" in df.columns
    assert "ANOTHER_EXTRA" in df.columns
    assert df.loc[0, "EXTRA_COLUMN"] == "extra_value"
    assert df.loc[0, "ANOTHER_EXTRA"] == 42


def test_load_jsonl_nonexistent_file() -> None:
    """Test that loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_jsonl_to_dataframe(Path("/nonexistent/file.jsonl"))


def test_load_jsonl_invalid_json(tmp_path: Path) -> None:
    """Test that invalid JSON raises ValueError."""
    temp_file = tmp_path / "invalid.jsonl"
    temp_file.write_text('{"valid": "json"}\ninvalid json line\n{"another": "valid"}\n')

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        load_jsonl_to_dataframe(temp_file)


def test_df_id_stability(tmp_path: Path) -> None:
    """Test that df_id remains stable across operations."""
    sample_data = [
        {"UUID_STRING": "uuid-1", "GPS_LATITUDE": 37.7749},
        {"UUID_STRING": "uuid-2", "GPS_LATITUDE": 37.7849},
        {"UUID_STRING": "uuid-3", "GPS_LATITUDE": 37.7949},
    ]

    temp_file = tmp_path / "stable.jsonl"
    temp_file.write_text("\n".join(json.dumps(record) for record in sample_data) + "\n")

    df = load_jsonl_to_dataframe(temp_file)

    # df_id should be stable integer index
    assert list(df.index) == [0, 1, 2]
    assert df.index.name == DF_ID_COLUMN

    # Test that we can reference rows by df_id
    assert df.loc[1, SchemaColumns.UUID_STRING] == "uuid-2"

    # Test filtering by df_id
    filtered = df.loc[[0, 2]]
    assert len(filtered) == 2
    assert list(filtered.index) == [0, 2]
