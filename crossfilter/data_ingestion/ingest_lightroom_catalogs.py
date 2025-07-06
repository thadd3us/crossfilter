"""CLI program for ingesting Lightroom catalog files into SQLite database."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import typer
from tqdm import tqdm

from crossfilter.core.schema import SchemaColumns
from crossfilter.data_ingestion.lightroom_parser import LightroomParserConfig, load_lightroom_catalog_to_df
from crossfilter.data_ingestion.sqlite_utils import upsert_dataframe_to_sqlite

logger = logging.getLogger(__name__)


def find_lightroom_catalogs(base_dir: Path) -> List[Path]:
    """Find all Lightroom catalog files recursively in the base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise ValueError(f"Base directory is not a directory: {base_dir}")

    # Find both .lrcat files and .zip files (which might contain catalogs)
    lrcat_files = list(base_dir.rglob("*.lrcat"))
    zip_files = list(base_dir.rglob("*.zip"))
    
    all_catalog_files = lrcat_files + zip_files
    logger.info(f"Found {len(lrcat_files)} .lrcat files and {len(zip_files)} .zip files in {base_dir}")

    return all_catalog_files



def main(
    base_dir: Path = typer.Argument(
        ..., help="Directory to recursively search for Lightroom catalog files (.lrcat and .zip)"
    ),
    destination_sqlite_db: Path = typer.Argument(
        ..., help="SQLite database file to upsert into"
    ),
    destination_table: str = typer.Option("data", help="Database table name"),
    include_metadata: bool = typer.Option(True, help="Include camera and EXIF metadata"),
    include_keywords: bool = typer.Option(True, help="Include keyword tags"),
    include_collections: bool = typer.Option(True, help="Include collection/album information"),
    ignore_collections: str = typer.Option(
        "quick collection", 
        help="Comma-separated list of collection names to ignore (case-insensitive)"
    ),
) -> None:
    """Ingest Lightroom catalog files into SQLite database."""
    logging.basicConfig(level=logging.INFO)

    # Parse ignore_collections
    ignore_set = set()
    if ignore_collections:
        ignore_set = {col.strip().lower() for col in ignore_collections.split(",") if col.strip()}

    # Create parser config
    config = LightroomParserConfig(
        ignore_collections=ignore_set,
        include_metadata=include_metadata,
        include_keywords=include_keywords,
        include_collections=include_collections,
    )

    # Find all catalog files
    catalog_files = find_lightroom_catalogs(base_dir)

    if not catalog_files:
        logger.info("No Lightroom catalog files found")
        return

    # Process files sequentially (no parallelism for simplicity as requested)
    logger.info(f"Processing {len(catalog_files)} Lightroom catalog files...")
    dataframes = []
    
    for catalog_file in tqdm(catalog_files, desc="Processing catalogs"):
        df = load_lightroom_catalog_to_df(catalog_file, config)
        if not df.empty:
            dataframes.append(df)

    if not dataframes:
        logger.info("No valid data found in Lightroom catalog files")
        return

    # Concatenate all DataFrames
    logger.info(f"Concatenating {len(dataframes)} DataFrames...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates based on UUID
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=[SchemaColumns.UUID_STRING])
    final_count = len(combined_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate records")

    # Check for any remaining duplicate UUIDs (should not happen)
    if combined_df[SchemaColumns.UUID_STRING].duplicated().any():
        logger.error("Duplicate UUIDs found in data after deduplication")
        duplicated_uuids = combined_df[combined_df[SchemaColumns.UUID_STRING].duplicated(keep=False)]
        logger.error(f"Duplicated UUIDs: {duplicated_uuids[SchemaColumns.UUID_STRING].unique()}")
        # Remove duplicates, keeping the first occurrence
        combined_df = combined_df.drop_duplicates(subset=[SchemaColumns.UUID_STRING], keep='first')

    logger.info(f"Total records: {len(combined_df)}")

    # Upsert to database
    upsert_dataframe_to_sqlite(combined_df, destination_sqlite_db, destination_table)

    logger.info("Lightroom catalog ingestion completed successfully")


if __name__ == "__main__":
    typer.run(main)