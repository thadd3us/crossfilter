"""Tests for schema module."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from crossfilter.core.schema import (
    DataType,
    TemporalLevel,
    get_h3_column_name,
    get_temporal_column_name,
    load_jsonl_to_dataframe,
    load_sqlite_to_dataframe,
)
from crossfilter.core.schema import (
    SchemaColumns as C,
)


def test_data_type_enum() -> None:
    """Test DataType enum values."""
    assert DataType.PHOTO == "PHOTO"
    assert DataType.VIDEO == "VIDEO"
    assert DataType.GPX_TRACKPOINT == "GPX_TRACKPOINT"
    assert DataType.GPX_WAYPOINT == "GPX_WAYPOINT"


def test_h3_column_name_construction() -> None:
    """Test H3 column name construction."""
    # Test the helper function
    assert get_h3_column_name(7) == "BUCKETED_H3_L7"
    assert get_h3_column_name(0) == "BUCKETED_H3_L0"
    assert get_h3_column_name(15) == "BUCKETED_H3_L15"

    # Test that it raises for invalid levels
    with pytest.raises(ValueError):
        get_h3_column_name(-1)
    with pytest.raises(ValueError):
        get_h3_column_name(16)


def test_temporal_column_name_construction() -> None:
    """Test temporal column name construction."""
    # Test that the temporal columns are constructed using TemporalLevel enum
    assert get_temporal_column_name(TemporalLevel.HOUR) == "BUCKETED_TIMESTAMP_HOUR"
    assert get_temporal_column_name(TemporalLevel.SECOND) == "BUCKETED_TIMESTAMP_SECOND"
    assert get_temporal_column_name(TemporalLevel.YEAR) == "BUCKETED_TIMESTAMP_YEAR"


def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    """Test loading an empty JSONL file."""
    temp_file = tmp_path / "empty.jsonl"
    temp_file.write_text("")

    df = load_jsonl_to_dataframe(temp_file)
    assert len(df) == 0
    assert df.index.name == C.DF_ID
    # Should have all required schema columns (COUNT is optional)
    required_columns = [
        C.UUID_STRING,
        C.DATA_TYPE,
        C.NAME,
        C.CAPTION,
        C.SOURCE_FILE,
        C.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
        C.TIMESTAMP_UTC,
        C.GPS_LATITUDE,
        C.GPS_LONGITUDE,
        C.RATING_0_TO_5,
        C.SIZE_IN_BYTES,
    ]
    for col in required_columns:
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
    assert df.index.name == C.DF_ID
    assert list(df.index) == [0, 1]  # df_id should be 0, 1

    # Check specific values
    assert df.loc[0, C.UUID_STRING] == "test-uuid-1"
    assert df.loc[0, C.GPS_LATITUDE] == 37.7749
    assert df.loc[1, C.DATA_TYPE] == "GPX_TRACKPOINT"

    # Check timestamp conversion
    assert isinstance(df[C.TIMESTAMP_UTC].dtype, pd.DatetimeTZDtype)


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

    # Check that all required schema columns exist (COUNT is optional)
    required_columns = [
        C.UUID_STRING,
        C.DATA_TYPE,
        C.NAME,
        C.CAPTION,
        C.SOURCE_FILE,
        C.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
        C.TIMESTAMP_UTC,
        C.GPS_LATITUDE,
        C.GPS_LONGITUDE,
        C.RATING_0_TO_5,
        C.SIZE_IN_BYTES,
    ]
    for col in required_columns:
        assert col in df.columns

    # Missing columns should be null except the ones we provided
    assert df.loc[0, C.UUID_STRING] == "test-uuid-1"
    assert pd.isna(df.loc[0, C.NAME])
    assert pd.isna(df.loc[0, C.TIMESTAMP_UTC])


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


def test_count_column_optional(tmp_path: Path) -> None:
    """Test that COUNT column is truly optional."""
    # Test data without COUNT column
    sample_data = [
        {
            "UUID_STRING": "test-uuid-1",
            "GPS_LATITUDE": 37.7749,
            "GPS_LONGITUDE": -122.4194,
        }
    ]

    temp_file = tmp_path / "no_count.jsonl"
    temp_file.write_text("\n".join(json.dumps(record) for record in sample_data) + "\n")

    df = load_jsonl_to_dataframe(temp_file)

    # COUNT should not be in columns since it wasn't provided
    assert C.COUNT not in df.columns

    # Test data with COUNT column
    sample_data_with_count = [
        {
            "UUID_STRING": "test-uuid-2",
            "GPS_LATITUDE": 37.7849,
            "GPS_LONGITUDE": -122.4094,
            "COUNT": 5,
        }
    ]

    temp_file_with_count = tmp_path / "with_count.jsonl"
    temp_file_with_count.write_text(
        "\n".join(json.dumps(record) for record in sample_data_with_count) + "\n"
    )

    df_with_count = load_jsonl_to_dataframe(temp_file_with_count)

    # COUNT should be present and have the correct value
    assert C.COUNT in df_with_count.columns
    assert df_with_count.loc[0, C.COUNT] == 5


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
    assert df.index.name == C.DF_ID

    # Test that we can reference rows by df_id
    assert df.loc[1, C.UUID_STRING] == "uuid-2"

    # Test filtering by df_id
    filtered = df.loc[[0, 2]]
    assert len(filtered) == 2
    assert list(filtered.index) == [0, 2]


@pytest.mark.skipif(os.environ.get("THAD_DATA_AVAILAVLE") != "true", reason="Data is only on Thad's laptop")
def test_thad_load_real_data() -> None:
    df = load_sqlite_to_dataframe(Path("~/data.sqlite").expanduser(), "data")
    assert df[C.UUID_STRING].isna().sum() == 0
    assert df[C.UUID_STRING].notna
    assert df[C.DATA_TYPE].isna().sum() == 0
    assert df[C.TIMESTAMP_UTC].isna().sum() == 0
    assert df[C.TIMESTAMP_UTC].dtype == pd.DatetimeTZDtype(tz="UTC")
