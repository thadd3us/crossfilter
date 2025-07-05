"""Tests for GPX files ingestion CLI program."""

import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.ingest_gpx_files import (
    create_or_update_table,
    find_gpx_files,
    get_pandas_to_sqlalchemy_dtype,
    process_single_gpx_file,
    upsert_dataframe_to_sqlite,
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


def test_get_pandas_to_sqlalchemy_dtype() -> None:
    """Test mapping pandas dtypes to SQLAlchemy types."""
    assert get_pandas_to_sqlalchemy_dtype("int64") == "INTEGER"
    assert get_pandas_to_sqlalchemy_dtype("float64") == "REAL"
    assert get_pandas_to_sqlalchemy_dtype("datetime64[ns, UTC]") == "TIMESTAMP"
    assert get_pandas_to_sqlalchemy_dtype("bool") == "BOOLEAN"
    assert get_pandas_to_sqlalchemy_dtype("object") == "TEXT"


def test_create_or_update_table_new_table(tmp_path: Path) -> None:
    """Test creating a new table."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create test DataFrame
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749],
        SchemaColumns.GPS_LONGITUDE: [-122.4194],
    })
    
    create_or_update_table(engine, "test_table", df)
    
    # Check table exists
    inspector = inspect(engine)
    assert inspector.has_table("test_table")
    
    # Check columns
    columns = {col["name"] for col in inspector.get_columns("test_table")}
    assert SchemaColumns.UUID_STRING in columns
    assert SchemaColumns.DATA_TYPE in columns
    assert SchemaColumns.GPS_LATITUDE in columns
    assert SchemaColumns.GPS_LONGITUDE in columns


def test_create_or_update_table_add_columns(tmp_path: Path) -> None:
    """Test adding columns to existing table."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create initial table with fewer columns
    initial_df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
    })
    
    create_or_update_table(engine, "test_table", initial_df)
    
    # Add new columns
    extended_df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid-2"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749],
        SchemaColumns.GPS_LONGITUDE: [-122.4194],
    })
    
    create_or_update_table(engine, "test_table", extended_df)
    
    # Check all columns exist
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("test_table")}
    assert SchemaColumns.UUID_STRING in columns
    assert SchemaColumns.DATA_TYPE in columns
    assert SchemaColumns.GPS_LATITUDE in columns
    assert SchemaColumns.GPS_LONGITUDE in columns


def test_create_or_update_table_nullable_columns(tmp_path: Path) -> None:
    """Test handling nullable columns that exist in table but not in DataFrame."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create table with more columns than DataFrame
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE test_table (
                UUID_STRING TEXT,
                DATA_TYPE TEXT,
                GPS_LATITUDE REAL,
                GPS_LONGITUDE REAL,
                EXTRA_COLUMN TEXT
            )
        """))
        conn.commit()
    
    # Create DataFrame with fewer columns
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749],
    })
    
    # Should not raise error when table has extra columns
    create_or_update_table(engine, "test_table", df)
    
    # Check table still exists with all columns
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("test_table")}
    assert "EXTRA_COLUMN" in columns
    assert SchemaColumns.GPS_LATITUDE in columns


def test_upsert_dataframe_to_sqlite_empty(tmp_path: Path) -> None:
    """Test upserting empty DataFrame."""
    db_path = tmp_path / "test.db"
    empty_df = pd.DataFrame()
    
    upsert_dataframe_to_sqlite(empty_df, db_path, "test_table")
    
    # Database should not be created for empty DataFrame
    assert not db_path.exists()


def test_upsert_dataframe_to_sqlite_new_table(tmp_path: Path) -> None:
    """Test upserting to new table."""
    db_path = tmp_path / "test.db"
    
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-2"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749, 37.7750],
        SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4195],
    })
    
    upsert_dataframe_to_sqlite(df, db_path, "test_table")
    
    # Check data was inserted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM test_table")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 2


def test_upsert_dataframe_to_sqlite_update_existing(tmp_path: Path) -> None:
    """Test upserting to existing table with updates."""
    db_path = tmp_path / "test.db"
    
    # Insert initial data
    initial_df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-2"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749, 37.7750],
        SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4195],
    })
    
    upsert_dataframe_to_sqlite(initial_df, db_path, "test_table")
    
    # Update with new data (one existing UUID, one new)
    update_df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-3"],  # uuid-1 exists, uuid-3 is new
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_TRACKPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7751, 37.7752],  # Updated latitude for uuid-1
        SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4196],
    })
    
    upsert_dataframe_to_sqlite(update_df, db_path, "test_table")
    
    # Check results
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Should have 3 total records (2 original + 1 new)
    cursor.execute("SELECT COUNT(*) FROM test_table")
    count = cursor.fetchone()[0]
    assert count == 3
    
    # Check that uuid-1 was updated
    cursor.execute("SELECT GPS_LATITUDE FROM test_table WHERE UUID_STRING = 'test-uuid-1'")
    updated_lat = cursor.fetchone()[0]
    assert updated_lat == 37.7751
    
    # Check that uuid-2 still exists unchanged
    cursor.execute("SELECT GPS_LATITUDE FROM test_table WHERE UUID_STRING = 'test-uuid-2'")
    original_lat = cursor.fetchone()[0]
    assert original_lat == 37.7750
    
    conn.close()


def test_upsert_dataframe_to_sqlite_with_none_values(tmp_path: Path) -> None:
    """Test upserting DataFrame with None values."""
    db_path = tmp_path / "test.db"
    
    df = pd.DataFrame({
        SchemaColumns.UUID_STRING: ["test-uuid-1"],
        SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
        SchemaColumns.GPS_LATITUDE: [37.7749],
        SchemaColumns.GPS_LONGITUDE: [-122.4194],
        SchemaColumns.NAME: [None],  # None value
        SchemaColumns.CAPTION: [None],  # None value
    })
    
    upsert_dataframe_to_sqlite(df, db_path, "test_table")
    
    # Check data was inserted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM test_table")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 1