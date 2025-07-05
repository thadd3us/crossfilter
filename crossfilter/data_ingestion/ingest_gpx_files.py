"""CLI program for ingesting GPX files into SQLite database."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import typer
from sqlalchemy import Column, MetaData, String, Table, create_engine, inspect, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import sessionmaker
from tqdm.contrib.concurrent import process_map
from crossfilter.core.schema import SchemaColumns as C

from crossfilter.core.schema import SchemaColumns
from crossfilter.data_ingestion.gpx_parser import load_gpx_file_to_df

logger = logging.getLogger(__name__)


def find_gpx_files(base_dir: Path) -> List[Path]:
    """Find all GPX files recursively in the base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise ValueError(f"Base directory is not a directory: {base_dir}")

    gpx_files = list(base_dir.rglob("*.gpx"))
    logger.info(f"Found {len(gpx_files)} GPX files in {base_dir}")

    return gpx_files


def process_single_gpx_file(gpx_file: Path) -> pd.DataFrame:
    """Process a single GPX file and return a DataFrame."""
    try:
        return load_gpx_file_to_df(gpx_file)
    except Exception as e:
        logger.error(f"Failed to process {gpx_file}: {e}")
        # Return empty DataFrame with correct schema on error
        return pd.DataFrame(
            columns=[
                SchemaColumns.UUID_STRING,
                SchemaColumns.DATA_TYPE,
                SchemaColumns.NAME,
                SchemaColumns.CAPTION,
                SchemaColumns.SOURCE_FILE,
                SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
                SchemaColumns.TIMESTAMP_UTC,
                SchemaColumns.GPS_LATITUDE,
                SchemaColumns.GPS_LONGITUDE,
                SchemaColumns.RATING_0_TO_5,
                SchemaColumns.SIZE_IN_BYTES,
            ]
        )


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


def create_or_update_table(engine, table_name: str, df: pd.DataFrame) -> None:
    """Create table or add missing columns if table exists."""
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
            # Try to create unique index, but handle the case where duplicates exist
            try:
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
            except Exception as e:
                if "UNIQUE constraint failed" in str(e):
                    logger.warning(
                        f"Cannot create unique index on {table_name}.{SchemaColumns.UUID_STRING} due to existing duplicates. Using fallback upsert strategy."
                    )
                    return False
                else:
                    # Re-raise other errors
                    raise e

        return has_unique


def upsert_dataframe_to_sqlite(
    df: pd.DataFrame, destination_sqlite_db: Path, destination_table: str
) -> None:
    """Upsert DataFrame to SQLite database using ON CONFLICT DO UPDATE or fallback strategy."""
    if df.empty:
        logger.info("No data to upsert")
        return

    # Create database directory if it doesn't exist
    destination_sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f"sqlite:///{destination_sqlite_db}")

    # Create or update table structure
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

            # Use INSERT OR REPLACE for SQLite upsert
            stmt = insert(table).values(cleaned_record)

            # For SQLite, we'll use ON CONFLICT DO UPDATE
            # First, let's check if UUID_STRING is the primary key
            primary_key_col = SchemaColumns.UUID_STRING

            # Create the ON CONFLICT DO UPDATE statement
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name != primary_key_col
            }

            if update_dict:
                stmt = stmt.on_conflict_do_update(
                    index_elements=[primary_key_col], set_=update_dict
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=[primary_key_col])

            conn.execute(stmt)

        conn.commit()

    logger.info(
        f"Upserted {len(records)} records to {destination_table} in {destination_sqlite_db}"
    )


def main(
    base_dir: Path = typer.Argument(
        ..., help="Directory to recursively search for GPX files"
    ),
    destination_sqlite_db: Path = typer.Argument(
        ..., help="SQLite database file to upsert into"
    ),
    destination_table: str = typer.Option("data", help="Database table name"),
    max_workers: int = typer.Option(None, help="Maximum number of parallel workers"),
) -> None:
    """Ingest GPX files into SQLite database."""
    logging.basicConfig(level=logging.INFO)

    # Find all GPX files
    gpx_files = find_gpx_files(base_dir)

    if not gpx_files:
        logger.info("No GPX files found")
        return

    # Process files in parallel
    logger.info(f"Processing {len(gpx_files)} GPX files...")
    dataframes = process_map(
        process_single_gpx_file,
        gpx_files,
        max_workers=max_workers,
        desc="Processing GPX files",
        chunksize=1,
    )

    # Filter out empty DataFrames
    non_empty_dataframes = [df for df in dataframes if not df.empty]

    if not non_empty_dataframes:
        logger.info("No valid data found in GPX files")
        return

    # Concatenate all DataFrames
    logger.info(f"Concatenating {len(non_empty_dataframes)} DataFrames...")
    combined_df = pd.concat(non_empty_dataframes, ignore_index=True)

    combined_df = combined_df.drop_duplicates()

    if combined_df[C.UUID_STRING].duplicated().any():
        logger.warning("Duplicate UUIDs found in data")
        combined_df = combined_df.drop_duplicates(subset=[C.UUID_STRING])

    logger.info(f"Total records: {len(combined_df)}")

    # Upsert to database
    upsert_dataframe_to_sqlite(combined_df, destination_sqlite_db, destination_table)

    logger.info("GPX ingestion completed successfully")


if __name__ == "__main__":
    typer.run(main)
