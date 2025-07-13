"""CLI program for ingesting GPX files into SQLite database."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import typer
from tqdm.contrib.concurrent import process_map
from crossfilter.core.schema import SchemaColumns as C

from crossfilter.core.schema import SchemaColumns
from crossfilter.data_ingestion.gpx.gpx_parser import load_gpx_file_to_df
from crossfilter.data_ingestion.sqlite_utils import upsert_dataframe_to_sqlite

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
        # assert False
        combined_df = combined_df.drop_duplicates(subset=[C.UUID_STRING])

    logger.info(
        f"Total records: {len(combined_df)} (H3 columns computed per-file in parallel)"
    )

    # Upsert to database
    upsert_dataframe_to_sqlite(combined_df, destination_sqlite_db, destination_table)

    logger.info("GPX ingestion completed successfully")


if __name__ == "__main__":
    typer.run(main)
