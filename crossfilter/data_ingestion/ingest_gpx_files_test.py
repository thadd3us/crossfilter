"""Tests for GPX files ingestion CLI program."""

import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.ingest_gpx_files import (
    find_gpx_files,
    process_single_gpx_file,
)


def create_test_gpx_file(path: Path, content: str) -> None:
    """Create a test GPX file with given content."""
    path.write_text(content)


def test_find_gpx_files_nonexistent_directory() -> None:
    """Test finding GPX files in non-existent directory."""
    nonexistent_dir = Path("/nonexistent/directory")
    
    with pytest.raises(FileNotFoundError):
        find_gpx_files(nonexistent_dir)


def test_find_gpx_files_not_directory(tmp_path: Path) -> None:
    """Test finding GPX files when path is not a directory."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    with pytest.raises(ValueError):
        find_gpx_files(test_file)


def test_find_gpx_files_empty_directory(tmp_path: Path) -> None:
    """Test finding GPX files in empty directory."""
    gpx_files = find_gpx_files(tmp_path)
    assert gpx_files == []


def test_find_gpx_files_with_files(tmp_path: Path) -> None:
    """Test finding GPX files in directory with files."""
    # Create some GPX files
    (tmp_path / "file1.gpx").write_text("content1")
    (tmp_path / "file2.gpx").write_text("content2")
    (tmp_path / "other.txt").write_text("content3")
    
    # Create subdirectory with GPX file
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.gpx").write_text("content4")
    
    gpx_files = find_gpx_files(tmp_path)
    
    assert len(gpx_files) == 3
    assert tmp_path / "file1.gpx" in gpx_files
    assert tmp_path / "file2.gpx" in gpx_files
    assert subdir / "file3.gpx" in gpx_files


def test_process_single_gpx_file_valid(tmp_path: Path) -> None:
    """Test processing a valid GPX file."""
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
    <trk>
        <name>Test Track</name>
        <trkseg>
            <trkpt lat="37.7749" lon="-122.4194">
                <time>2023-01-01T12:00:00Z</time>
            </trkpt>
        </trkseg>
    </trk>
</gpx>"""
    
    gpx_file = tmp_path / "test.gpx"
    create_test_gpx_file(gpx_file, gpx_content)
    
    df = process_single_gpx_file(gpx_file)
    
    assert len(df) == 1
    assert df.iloc[0][SchemaColumns.DATA_TYPE] == DataType.GPX_TRACKPOINT
    assert df.iloc[0][SchemaColumns.GPS_LATITUDE] == 37.7749
    assert df.iloc[0][SchemaColumns.GPS_LONGITUDE] == -122.4194


def test_process_single_gpx_file_invalid(tmp_path: Path) -> None:
    """Test processing an invalid GPX file."""
    invalid_file = tmp_path / "invalid.gpx"
    invalid_file.write_text("This is not a valid GPX file")
    
    df = process_single_gpx_file(invalid_file)
    
    # Should return empty DataFrame with correct schema
    assert len(df) == 0
    assert SchemaColumns.UUID_STRING in df.columns
    assert SchemaColumns.DATA_TYPE in df.columns


