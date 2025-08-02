"""Tests for shared SQLite utility functions."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy.exc
from sqlalchemy import Engine, create_engine, text
from syrupy.assertion import SnapshotAssertion

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.sqlite_utils import (
    create_or_update_table_schema,
    get_pandas_to_sqlalchemy_dtype,
    query_sqlite_to_dataframe,
    upsert_dataframe_to_sqlite,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    """Create a test engine."""
    db_path = tmp_path / "test.db"
    return create_engine(f"sqlite:///{db_path}")


def test_get_pandas_to_sqlalchemy_dtype() -> None:
    """Test mapping pandas dtypes to SQLAlchemy types."""
    assert get_pandas_to_sqlalchemy_dtype("int64") == "INTEGER"
    assert get_pandas_to_sqlalchemy_dtype("float64") == "REAL"
    assert get_pandas_to_sqlalchemy_dtype("datetime64[ns, UTC]") == "TIMESTAMP"
    assert get_pandas_to_sqlalchemy_dtype("bool") == "BOOLEAN"
    assert get_pandas_to_sqlalchemy_dtype("object") == "TEXT"


def test_create_or_update_table_new_table(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test creating a new table."""
    # Create test DataFrame
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749],
            SchemaColumns.GPS_LONGITUDE: [-122.4194],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)
    create_or_update_table_schema(engine, "test_table", df)
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_create_or_update_table_add_columns(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test adding columns to existing table."""
    initial_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
        }
    )
    initial_df = initial_df.set_index(SchemaColumns.UUID_STRING)
    create_or_update_table_schema(engine, "test_table", initial_df)
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot

    extended_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid-2"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749],
            SchemaColumns.GPS_LONGITUDE: [-122.4194],
        }
    )
    extended_df = extended_df.set_index(SchemaColumns.UUID_STRING)
    create_or_update_table_schema(engine, "test_table", extended_df)
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_create_or_update_table_nullable_columns(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test handling nullable columns that exist in table but not in DataFrame."""

    # Create table with more columns than DataFrame
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE test_table (
                UUID_STRING TEXT,
                DATA_TYPE TEXT,
                GPS_LATITUDE REAL,
                GPS_LONGITUDE REAL,
                EXTRA_COLUMN TEXT
            )
        """
            )
        )
        conn.commit()
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot

    # Create DataFrame with fewer columns
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)
    create_or_update_table_schema(engine, "test_table", df)
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_upsert_dataframe_to_sqlite_empty(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test upserting empty DataFrame."""
    db_path = tmp_path / "test.db"
    empty_df = pd.DataFrame()
    empty_df.index.name = SchemaColumns.UUID_STRING

    with pytest.raises(ValueError):
        upsert_dataframe_to_sqlite(empty_df, db_path, "test_table")


def test_upsert_dataframe_to_sqlite_new_table(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test upserting to new table."""

    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-2"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749, 37.7750],
            SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4195],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)
    upsert_dataframe_to_sqlite(df, engine, "test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_upsert_dataframe_to_sqlite_update_existing(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test upserting to existing table with updates."""
    # Insert initial data
    initial_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-2"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
            SchemaColumns.GPS_LATITUDE: [1.1, 2.2],
            SchemaColumns.GPS_LONGITUDE: [-10.1, -20.2],
        }
    )
    initial_df = initial_df.set_index(SchemaColumns.UUID_STRING)
    upsert_dataframe_to_sqlite(initial_df, engine, "test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot

    # Update with new data (one existing UUID, one new)
    update_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: [
                "test-uuid-1",
                "test-uuid-3",
            ],  # uuid-1 exists, uuid-3 is new
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [1.2, 3.3],  # Updated latitude for uuid-1
            SchemaColumns.GPS_LONGITUDE: [-10.2, -30.3],
        }
    )
    update_df = update_df.set_index(SchemaColumns.UUID_STRING)
    upsert_dataframe_to_sqlite(update_df, engine, "test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_upsert_dataframe_to_sqlite_with_none_values(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test upserting DataFrame with None values."""
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid-1"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749],
            SchemaColumns.GPS_LONGITUDE: [-122.4194],
            SchemaColumns.NAME: [None],  # None value
            SchemaColumns.CAPTION: [None],  # None value
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)

    upsert_dataframe_to_sqlite(df, engine, "test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual.to_dict(orient="records") == snapshot


def test_existing_table_without_unique_constraint(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test handling of existing table without unique constraint on UUID_STRING."""
    # Create a table manually without unique constraint (simulating old database)
    initial_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid-1", "uuid-2"],
            SchemaColumns.DATA_TYPE: [
                DataType.GPX_TRACKPOINT,
                DataType.GPX_TRACKPOINT,
            ],
            SchemaColumns.GPS_LATITUDE: [37, 38],
            SchemaColumns.GPS_LONGITUDE: [-122, -123],
        }
    )
    initial_df.to_sql("data", engine, if_exists="replace", index=False)

    # Now try to update the table structure and upsert new data
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid-2", "uuid-3"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [39, 40],
            SchemaColumns.GPS_LONGITUDE: [-124, -125],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)
    # This should work - the function should handle missing unique constraint
    upsert_dataframe_to_sqlite(df, engine, "data")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM data")
    assert actual.to_dict(orient="records") == snapshot


def test_existing_table_with_duplicate_uuids(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test that attempting to create unique constraint on table with duplicates raises an exception."""

    initial_df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid-1", "uuid-1"],
            SchemaColumns.DATA_TYPE: [
                DataType.GPX_TRACKPOINT,
                DataType.GPX_TRACKPOINT,
            ],
            SchemaColumns.GPS_LATITUDE: [37, 38],
            SchemaColumns.GPS_LONGITUDE: [-122, -123],
        }
    )
    initial_df.to_sql("data", engine, if_exists="replace", index=False)

    # Try to upsert new data
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["new-uuid"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7751],
            SchemaColumns.GPS_LONGITUDE: [-122.4196],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)

    # This should raise an exception when trying to create unique constraint
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        upsert_dataframe_to_sqlite(df, engine, "data")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM data")
    assert actual.to_dict(orient="records") == snapshot


def test_upsert_with_clean_data_works_properly(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test that upsert works properly when data doesn't have duplicate UUIDs."""
    # First, insert some clean data
    df1 = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid-1", "uuid-2"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_TRACKPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749, 37.7750],
            SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4195],
        }
    )
    df1 = df1.set_index(SchemaColumns.UUID_STRING)

    upsert_dataframe_to_sqlite(df1, engine, "data")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM data")
    assert actual.to_dict(orient="records") == snapshot

    # Now try to upsert data that updates existing record and adds new one
    df2 = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: [
                "uuid-1",
                "uuid-3",
            ],  # Update uuid-1, add uuid-3
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
            SchemaColumns.GPS_LATITUDE: [
                37.7799,
                37.7800,
            ],  # Updated latitude for uuid-1
            SchemaColumns.GPS_LONGITUDE: [-122.4200, -122.4201],
        }
    )
    df2 = df2.set_index(SchemaColumns.UUID_STRING)

    upsert_dataframe_to_sqlite(df2, engine, "data")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM data")
    assert actual.to_dict(orient="records") == snapshot


def test_query_sqlite_to_dataframe_with_params(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test querying SQLite database with parameters."""
    # Create test data
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["test-uuid-1", "test-uuid-2"],
            SchemaColumns.DATA_TYPE: [DataType.GPX_TRACKPOINT, DataType.GPX_WAYPOINT],
            SchemaColumns.GPS_LATITUDE: [37.7749, 37.7750],
            SchemaColumns.GPS_LONGITUDE: [-122.4194, -122.4195],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)

    upsert_dataframe_to_sqlite(df, engine, "test_table")

    # Query with parameters
    result_df = query_sqlite_to_dataframe(
        engine,
        f"SELECT * FROM test_table WHERE {SchemaColumns.UUID_STRING} = :uuid",
        {"uuid": "test-uuid-1"},
    )
    assert result_df.to_dict(orient="records") == snapshot


def test_upsert_large_dataframe(engine: Engine) -> None:
    """Test upserting a large dataframe."""
    # Create a large dataframe
    df = pd.DataFrame()
    df[str(SchemaColumns.UUID_STRING)] = [f"uuid_{i}" for i in range(10_000)]
    for col in range(20):
        df[f"col_{col}"] = [f"value_{col}_{i}" for i in range(len(df))]
    df = df.set_index(SchemaColumns.UUID_STRING)

    logger.info(f"Upserting {len(df)} rows to test_table")
    upsert_dataframe_to_sqlite(df, engine, "test_table")
    logger.info(f"Upserted {len(df)} rows to test_table")

    logger.info(f"Querying {len(df)} rows from test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    logger.info(f"Queried {len(actual)} rows from test_table")

    assert actual.columns.tolist() == [df.index.name] + df.columns.tolist()
    assert actual[df.columns].values.tolist() == df.values.tolist()


def test_thad_upsert_datetime_utc_is_preserved(
    engine: Engine, snapshot: SnapshotAssertion
) -> None:
    """Test that upserting a dataframe with a datetime column preserves the UTC timezone."""
    df = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid-1"],
            SchemaColumns.TIMESTAMP_UTC: [
                pd.Timestamp("2021-01-01 12:00:00", tz="UTC")
            ],
        }
    )
    df = df.set_index(SchemaColumns.UUID_STRING)
    upsert_dataframe_to_sqlite(df, engine, "test_table")
    actual = query_sqlite_to_dataframe(engine, "SELECT * FROM test_table")
    assert actual[SchemaColumns.TIMESTAMP_UTC].dtype == pd.DatetimeTZDtype(tz="UTC")
    assert actual.dtypes.to_dict() == snapshot
    assert actual.to_dict(orient="records") == snapshot
