"""Tests for GPX file parsing functions."""

import uuid
from datetime import datetime
from pathlib import Path

from syrupy.assertion import SnapshotAssertion
import pytest

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.gpx_parser import (
    generate_uuid_from_components,
    load_gpx_file_to_df,
)


def test_generate_uuid_from_components() -> None:
    """Test UUID generation from components."""
    timestamp1 = datetime(2023, 1, 1, 12, 0, 0)
    timestamp2 = datetime(2023, 1, 1, 12, 5, 0)

    # Same inputs should generate the same UUID
    uuid1 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7750,
        -122.4195,
        DataType.GPX_TRACKPOINT,
    )
    uuid2 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7750,
        -122.4195,
        DataType.GPX_TRACKPOINT,
    )

    assert uuid1 == uuid2
    assert uuid.UUID(uuid1)  # Should be valid UUID

    # Different inputs should generate different UUIDs
    uuid3 = generate_uuid_from_components(
        timestamp1,
        37.7749,
        -122.4194,
        timestamp2,
        37.7750,
        -122.4195,
        DataType.GPX_WAYPOINT,
    )

    assert uuid1 != uuid3


def test_load_gpx_file_to_df_nonexistent_file() -> None:
    """Test loading a non-existent GPX file."""
    nonexistent_file = Path("/nonexistent/file.gpx")

    with pytest.raises(FileNotFoundError):
        load_gpx_file_to_df(nonexistent_file)


def test_load_gpx_file_to_df_valid_file(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a valid GPX file."""
    # Create a simple GPX file
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
    <trk>
        <name>Test Track</name>
        <trkseg>
            <trkpt lat="37.7749" lon="-122.4194">
                <time>2023-01-01T12:00:00Z</time>
            </trkpt>
            <trkpt lat="37.7750" lon="-122.4195">
                <time>2023-01-01T12:05:00Z</time>
            </trkpt>
        </trkseg>
    </trk>
    <wpt lat="37.7751" lon="-122.4196">
        <name>Test Waypoint</name>
        <desc>A test waypoint</desc>
        <time>2023-01-01T12:10:00Z</time>
    </wpt>
</gpx>"""

    gpx_file = tmp_path / "test.gpx"
    gpx_file.write_text(gpx_content)

    df = load_gpx_file_to_df(gpx_file)
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot


def test_load_gpx_file_to_df_empty_file(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading an empty GPX file."""
    # Create an empty GPX file
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
</gpx>"""

    gpx_file = tmp_path / "empty.gpx"
    gpx_file.write_text(gpx_content)

    df = load_gpx_file_to_df(gpx_file)
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot


def test_load_gpx_file_to_df_invalid_file(tmp_path: Path) -> None:
    """Test loading an invalid GPX file."""
    invalid_file = tmp_path / "invalid.gpx"
    invalid_file.write_text("This is not a valid GPX file")

    with pytest.raises(ValueError, match="Failed to parse GPX file"):
        load_gpx_file_to_df(invalid_file)


def test_load_gpx_file_to_df_trackpoints_without_time(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading a GPX file with trackpoints without timestamps."""
    # Create a GPX file with trackpoints but no timestamps
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
    <trk>
        <name>Test Track</name>
        <trkseg>
            <trkpt lat="37.7749" lon="-122.4194">
            </trkpt>
            <trkpt lat="37.7750" lon="-122.4195">
            </trkpt>
        </trkseg>
    </trk>
</gpx>"""

    gpx_file = tmp_path / "no_time.gpx"
    gpx_file.write_text(gpx_content)

    df = load_gpx_file_to_df(gpx_file)
    assert list(zip(df.columns, df.dtypes)) == snapshot
    assert df.to_dict(orient="records") == snapshot
