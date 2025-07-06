"""Shared SQLite utility functions for data ingestion."""

import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import Column, MetaData, String, Table, create_engine, inspect, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import sessionmaker

from crossfilter.core.schema import SchemaColumns

logger = logging.getLogger(__name__)


def get_pandas_to_sqlalchemy_dtype(pandas_dtype: str) -> str:
    """Map pandas dtype to SQLAlchemy type string."""
    if pandas_dtype.startswith("int"):
        return "INTEGER"
    elif pandas_dtype.startswith("float"):
        return "REAL"
    elif pandas_dtype.startswith("datetime"):
        return "TIMESTAMP"
    elif pandas_dtype.startswith("bool"):
        return "BOOLEAN"
    else:
        return "TEXT"


def has_unique_constraint_on_uuid(engine, table_name: str) -> bool:
    """Check if table has a unique constraint or index on UUID_STRING column."""
    inspector = inspect(engine)

    # Check for unique constraints
    try:
        unique_constraints = inspector.get_unique_constraints(table_name)
        for constraint in unique_constraints:
            if SchemaColumns.UUID_STRING in constraint.get("column_names", []):
                return True
    except Exception as e:
        logger.debug(f"get_unique_constraints not supported: {e}")

    # Check for unique indexes
    try:
        indexes = inspector.get_indexes(table_name)
        for index in indexes:
            if index.get("unique", False) and SchemaColumns.UUID_STRING in index.get(
                "column_names", []
            ):
                return True
    except Exception as e:
        logger.debug(f"get_indexes failed: {e}")

    return False


def create_or_update_table(engine, table_name: str, df: pd.DataFrame) -> bool:
    """Create table or add missing columns if table exists.

    Returns True if unique constraint exists on UUID_STRING, False otherwise.
    """
    inspector = inspect(engine)

    if not inspector.has_table(table_name):
        # Create new table
        logger.info(f"Creating new table: {table_name}")
        df.to_sql(table_name, engine, if_exists="replace", index=False)

        # Set UUID_STRING as primary key
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_uuid ON {table_name} ({SchemaColumns.UUID_STRING})"
                )
            )
            conn.commit()
        return True
    else:
        # Check if we need to add columns
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        df_columns = set(df.columns)
        missing_columns = df_columns - existing_columns

        if missing_columns:
            logger.info(f"Adding missing columns to {table_name}: {missing_columns}")

            with engine.connect() as conn:
                for col in missing_columns:
                    dtype = get_pandas_to_sqlalchemy_dtype(str(df[col].dtype))
                    conn.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype}")
                    )
                conn.commit()

        # Check if unique constraint exists on UUID_STRING
        has_unique = has_unique_constraint_on_uuid(engine, table_name)

        if not has_unique:
            # Create unique index - let exceptions propagate if there are duplicates
            with engine.connect() as conn:
                conn.execute(
                    text(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_uuid ON {table_name} ({SchemaColumns.UUID_STRING})"
                    )
                )
                conn.commit()
            logger.info(
                f"Created unique index on {SchemaColumns.UUID_STRING} for table {table_name}"
            )
            return True

        return has_unique


def upsert_dataframe_to_sqlite(
    df: pd.DataFrame, destination_sqlite_db: Path, destination_table: str
) -> None:
    """Upsert DataFrame to SQLite database using ON CONFLICT DO UPDATE."""
    if df.empty:
        logger.info("No data to upsert")
        return

    # Create database directory if it doesn't exist
    destination_sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f"sqlite:///{destination_sqlite_db}")

    # Create or update table structure - will ensure unique constraint exists
    create_or_update_table(engine, destination_table, df)

    # Perform upsert using SQLAlchemy
    metadata = MetaData()

    # Reflect the table structure
    table = Table(destination_table, metadata, autoload_with=engine)

    # Convert DataFrame to list of dictionaries
    records = df.to_dict("records")

    # Perform upsert
    with engine.connect() as conn:
        for record in records:
            # Clean None values for SQLite
            cleaned_record = {k: v for k, v in record.items() if v is not None}

            # Use ON CONFLICT DO UPDATE
            stmt = insert(table).values(cleaned_record)
            primary_key_col = SchemaColumns.UUID_STRING

            # Create the ON CONFLICT DO UPDATE statement
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name != primary_key_col
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=[primary_key_col], set_=update_dict
            )
            conn.execute(stmt)

        conn.commit()

    logger.info(
        f"Upserted {len(records)} records to {destination_table} in {destination_sqlite_db}"
    )


def query_sqlite_to_dataframe(
    sqlite_db_path: Path, query: str, params: Dict[str, Any] = None
) -> pd.DataFrame:
    """Execute SQL query against SQLite database and return DataFrame."""
    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df
