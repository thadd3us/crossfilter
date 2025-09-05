"""Tests for Google Takeout location history parsing functions."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from crossfilter.data_ingestion.gpx.google_takeout_parser import (
    convert_e7_coordinate,
    generate_uuid_from_components,
    load_google_takeout_records_to_df,
    parse_location_record,
    parse_takeout_timestamp,
)
from crossfilter.core.schema import DataType


def create_sample_takeout_data() -> dict:
    """Create sample Google Takeout Records.json data for testing."""
    return {
        "locations": [
            {
                "timestamp": "2022-01-12T17:18:24.190Z",
                "latitudeE7": 377749000,  # 37.7749
                "longitudeE7": -1224194000,  # -122.4194
                "accuracy": 20,
                "source": "WIFI",
                "deviceTag": 1234567890,
                "platformType": "ANDROID"
            },
            {
                "timestamp": "2022-01-12T17:23:24.000Z",
                "latitudeE7": 377750000,  # 37.7750
                "longitudeE7": -1224195000,  # -122.4195
                "accuracy": 15,
                "source": "GPS",
                "altitude": 150
            },
            {
                "timestamp": "2022-01-12T17:28:24.500Z",
                "latitudeE7": 377751000,  # 37.7751
                "longitudeE7": -1224196000,  # -122.4196
                "accuracy": 25
            }
        ]
    }


def test_parse_takeout_timestamp() -> None:
    """Test parsing Google Takeout timestamp strings."""
    # Test standard ISO 8601 format with Z suffix
    timestamp_str = "2022-01-12T17:18:24.190Z"
    dt = parse_takeout_timestamp(timestamp_str)
    assert dt == datetime(2022, 1, 12, 17, 18, 24, 190000, tzinfo=timezone.utc)

    # Test timestamp without milliseconds
    timestamp_str = "2022-01-12T17:18:24Z"
    dt = parse_takeout_timestamp(timestamp_str)
    assert dt == datetime(2022, 1, 12, 17, 18, 24, 0, tzinfo=timezone.utc)

    # Test invalid timestamp
    with pytest.raises(ValueError, match="Failed to parse timestamp"):
        parse_takeout_timestamp("invalid-timestamp")


def test_convert_e7_coordinate() -> None:
    """Test converting Google's E7 coordinate format."""
    # Test latitude conversion (San Francisco)
    lat_e7 = 377749000  # 37.7749 degrees
    lat_decimal = convert_e7_coordinate(lat_e7)
    assert abs(lat_decimal - 37.7749) < 1e-7

    # Test longitude conversion (San Francisco)
    lon_e7 = -1224194000  # -122.4194 degrees
    lon_decimal = convert_e7_coordinate(lon_e7)
    assert abs(lon_decimal - (-122.4194)) < 1e-7

    # Test zero coordinate
    assert convert_e7_coordinate(0) == 0.0


def test_generate_uuid_from_components() -> None:
    """Test UUID generation from components."""
    timestamp1 = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    timestamp2 = datetime(2023, 1, 1, 12, 5, 0, tzinfo=timezone.utc)

    # Same inputs should generate the same UUID
    uuid1 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7750,
        -122.4195,
        DataType.GOOGLE_TAKEOUT_LOCATION,
    )
    uuid2 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7750,
        -122.4195,
        DataType.GOOGLE_TAKEOUT_LOCATION,
    )

    assert uuid1 == uuid2
    assert len(uuid1) == 16  # Should be 16 bytes

    # Different inputs should generate different UUIDs
    uuid3 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7751,  # Different latitude
        -122.4195,
        DataType.GOOGLE_TAKEOUT_LOCATION,
    )

    assert uuid1 != uuid3


def test_parse_location_record() -> None:
    """Test parsing individual location records."""
    # Valid record
    record = {
        "timestamp": "2022-01-12T17:18:24.190Z",
        "latitudeE7": 377749000,
        "longitudeE7": -1224194000,
        "accuracy": 20,
    }

    parsed = parse_location_record(record, "Records.json")
    assert parsed is not None
    assert parsed["DATA_TYPE"] == DataType.GOOGLE_TAKEOUT_LOCATION
    assert parsed["SOURCE_FILE"] == "Records.json"
    assert parsed["GPS_LATITUDE"] == 37.7749
    assert parsed["GPS_LONGITUDE"] == -122.4194

    # Record missing timestamp
    record_no_timestamp = {
        "latitudeE7": 377749000,
        "longitudeE7": -1224194000,
    }
    assert parse_location_record(record_no_timestamp, "Records.json") is None

    # Record missing coordinates
    record_no_coords = {
        "timestamp": "2022-01-12T17:18:24.190Z",
    }
    assert parse_location_record(record_no_coords, "Records.json") is None

    # Record with invalid latitude
    record_invalid_lat = {
        "timestamp": "2022-01-12T17:18:24.190Z",
        "latitudeE7": 1000000000,  # 100 degrees (invalid)
        "longitudeE7": -1224194000,
    }
    assert parse_location_record(record_invalid_lat, "Records.json") is None

    # Record with invalid longitude
    record_invalid_lon = {
        "timestamp": "2022-01-12T17:18:24.190Z",
        "latitudeE7": 377749000,
        "longitudeE7": -2000000000,  # -200 degrees (invalid)
    }
    assert parse_location_record(record_invalid_lon, "Records.json") is None


def test_load_google_takeout_records_to_df_nonexistent_file() -> None:
    """Test loading a non-existent Records.json file."""
    nonexistent_file = Path("/nonexistent/Records.json")

    with pytest.raises(FileNotFoundError):
        load_google_takeout_records_to_df(nonexistent_file)


def test_load_google_takeout_records_to_df_valid_file(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a valid Google Takeout Records.json file."""
    records_data = create_sample_takeout_data()

    records_file = tmp_path / "Records.json"
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(records_data, f)

    df = load_google_takeout_records_to_df(records_file)
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot


def test_load_google_takeout_records_to_df_empty_locations(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a Records.json file with empty locations array."""
    records_data = {"locations": []}

    records_file = tmp_path / "empty_Records.json"
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(records_data, f)

    df = load_google_takeout_records_to_df(records_file)
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot


def test_load_google_takeout_records_to_df_invalid_json(tmp_path: Path) -> None:
    """Test loading an invalid JSON file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("This is not valid JSON")

    with pytest.raises(ValueError, match="Failed to parse JSON file"):
        load_google_takeout_records_to_df(invalid_file)


def test_load_google_takeout_records_to_df_missing_locations_array(tmp_path: Path) -> None:
    """Test loading a JSON file without locations array."""
    invalid_data = {"some_other_key": "value"}

    invalid_file = tmp_path / "no_locations.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f)

    with pytest.raises(ValueError, match="missing 'locations' array"):
        load_google_takeout_records_to_df(invalid_file)


def test_load_google_takeout_records_to_df_locations_not_array(tmp_path: Path) -> None:
    """Test loading a JSON file where locations is not an array."""
    invalid_data = {"locations": "not an array"}

    invalid_file = tmp_path / "locations_not_array.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f)

    with pytest.raises(ValueError, match="'locations' is not an array"):
        load_google_takeout_records_to_df(invalid_file)


def test_load_google_takeout_records_to_df_mixed_valid_invalid_records(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a Records.json file with mix of valid and invalid records."""
    records_data = {
        "locations": [
            # Valid record
            {
                "timestamp": "2022-01-12T17:18:24.190Z",
                "latitudeE7": 377749000,
                "longitudeE7": -1224194000,
                "accuracy": 20
            },
            # Invalid record - missing timestamp
            {
                "latitudeE7": 377750000,
                "longitudeE7": -1224195000,
            },
            # Invalid record - missing coordinates
            {
                "timestamp": "2022-01-12T17:23:24.000Z",
                "accuracy": 15
            },
            # Valid record
            {
                "timestamp": "2022-01-12T17:28:24.500Z",
                "latitudeE7": 377751000,
                "longitudeE7": -1224196000,
                "accuracy": 25
            },
            # Invalid record - non-dict
            "not a dict",
            # Invalid record - bad coordinates
            {
                "timestamp": "2022-01-12T17:33:24.000Z",
                "latitudeE7": 1000000000,  # Invalid latitude (100 degrees)
                "longitudeE7": -1224196000,
            }
        ]
    }

    records_file = tmp_path / "mixed_Records.json"
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(records_data, f)

    df = load_google_takeout_records_to_df(records_file)
    # Should only load the 2 valid records
    assert len(df) == 2
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot


def test_load_google_takeout_records_to_df_all_invalid_records(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a Records.json file where all records are invalid."""
    records_data = {
        "locations": [
            # Missing timestamp
            {
                "latitudeE7": 377749000,
                "longitudeE7": -1224194000,
            },
            # Missing coordinates
            {
                "timestamp": "2022-01-12T17:18:24.190Z",
                "accuracy": 20
            },
            # Invalid coordinates
            {
                "timestamp": "2022-01-12T17:23:24.000Z",
                "latitudeE7": 1000000000,  # 100 degrees (invalid)
                "longitudeE7": -2000000000,  # -200 degrees (invalid)
            }
        ]
    }

    records_file = tmp_path / "all_invalid_Records.json"
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(records_data, f)

    df = load_google_takeout_records_to_df(records_file)
    assert len(df) == 0
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot