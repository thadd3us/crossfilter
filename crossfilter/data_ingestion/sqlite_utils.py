"""Shared SQLite utility functions for data ingestion."""

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Union

import pandas as pd
import tqdm
from sqlalchemy import (
    Engine,
    create_engine,
    inspect,
    text,
)

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


def has_unique_constraint_on_uuid(engine: Engine, table_name: str) -> bool:
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


def create_or_update_table_schema(
    engine: Engine, table_name: str, df: pd.DataFrame
) -> None:
    """Create table or add missing columns if table exists."""
    # Validate table name
    if not is_valid_sql_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    inspector = inspect(engine)
    with engine.connect() as conn:
        if not inspector.has_table(table_name):
            # Create new empty table
            logger.info(f"Creating new empty table: {table_name}")
            conn.execute(
                text(f"CREATE TABLE {table_name} ({SchemaColumns.UUID_STRING} STRING)")
            )

        # Check if we need to add columns
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        df_columns = set(df.columns)
        missing_columns = df_columns - existing_columns

        if missing_columns:
            logger.info(f"Adding missing columns to {table_name}: {missing_columns}")
            for col in missing_columns:
                if not is_valid_sql_identifier(col):
                    raise ValueError(f"Invalid column name: {col}")
                dtype = get_pandas_to_sqlalchemy_dtype(str(df[col].dtype))
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype}"))

        # Check if unique constraint exists on UUID_STRING
        has_unique = has_unique_constraint_on_uuid(engine, table_name)
        if not has_unique:
            # Create unique index - let exceptions propagate if there are duplicates
            conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_uuid ON {table_name} ({SchemaColumns.UUID_STRING})"
                )
            )
            logger.info(
                f"Created unique index on {SchemaColumns.UUID_STRING} for table {table_name}"
            )
        conn.commit()
    logger.info(f"Committed table {table_name} with schema {df.columns.tolist()}")


def upsert_dataframe_to_sqlite(
    df: pd.DataFrame, destination_sqlite_db: Path, destination_table: str
) -> None:
    """Upsert DataFrame to SQLite database using bulk operations."""
    if df.columns.empty:
        raise ValueError("DataFrame has no columns, we don't know what schema to use.")

    # Validate table name
    if not is_valid_sql_identifier(destination_table):
        raise ValueError(f"Invalid table name: {destination_table}")

    # Create database directory if it doesn't exist
    destination_sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f"sqlite:///{destination_sqlite_db}")

    # Create or update table structure - will ensure unique constraint exists
    create_or_update_table_schema(engine, destination_table, df)

    # For bulk upsert, we'll use a temporary table approach
    temp_table = f"{destination_table}_temp_{uuid.uuid4().hex}"

    with engine.connect() as conn:
        # Create temporary table with same structure
        # conn.execute(
        #     text(
        #         f"CREATE TABLE {escape_sql_identifier(temp_table)} AS SELECT * FROM {escape_sql_identifier(destination_table)} WHERE 1=0"
        #     )
        # )

        max_rows_per_insert = 500
        logger.info(
            f"Inserting {len(df)} rows to {temp_table} in batches of {max_rows_per_insert}"
        )
        for i in tqdm.tqdm(range(0, len(df), max_rows_per_insert)):
            df.iloc[i : i + max_rows_per_insert].to_sql(
                temp_table, engine, if_exists="append", index=False, method="multi"
            )
        logger.info(f"Upserted {len(df)} rows to {temp_table}")

        # Perform bulk upsert using a single INSERT OR REPLACE statement
        # This is much more efficient than individual row processing
        columns = list(df.columns)

        # Validate all column names
        for col in columns:
            if not is_valid_sql_identifier(col):
                raise ValueError(f"Invalid column name: {col}")

        columns_str = ", ".join(escape_sql_identifier(col) for col in columns)

        # Single bulk upsert operation
        upsert_stmt = text(
            f"""
            INSERT OR REPLACE INTO {escape_sql_identifier(destination_table)} ({columns_str})
            SELECT {columns_str} FROM {escape_sql_identifier(temp_table)}
        """
        )
        conn.execute(upsert_stmt)

        # Clean up temporary table
        conn.execute(text(f"DROP TABLE {temp_table}"))

        conn.commit()

    logger.info(
        f"Upserted {len(df)} records to {destination_table} in {destination_sqlite_db}"
    )


def query_sqlite_to_dataframe(
    sqlite_db_path: Path, query: str, params: Union[dict[str, Any], None] = None
) -> pd.DataFrame:
    """Execute SQL query against SQLite database and return DataFrame."""
    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # Workaround: https://github.com/pandas-dev/pandas/issues/55554
    if SchemaColumns.TIMESTAMP_UTC in df.columns:
        df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(
            df[SchemaColumns.TIMESTAMP_UTC], utc=True, format="ISO8601"
        )

    return df


def is_valid_sql_identifier(identifier: str) -> bool:
    """Validate that a string is a safe SQL identifier (column/table name)."""
    # Only allow alphanumeric characters, underscores, and hyphens
    # Must start with a letter or underscore
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_-]*$"
    return bool(re.match(pattern, identifier))


def escape_sql_identifier(identifier: str) -> str:
    """Escape a SQL identifier by wrapping in double quotes."""
    if not is_valid_sql_identifier(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return f'"{identifier}"'
