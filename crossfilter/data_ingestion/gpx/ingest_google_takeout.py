"""CLI program for ingesting Google Takeout Records.json files into parquet."""

import logging
from pathlib import Path

import pandas as pd
import typer
from tqdm.contrib.concurrent import process_map

from crossfilter.core.schema import SchemaColumns as C
from crossfilter.data_ingestion.gpx.google_takeout_parser import load_google_takeout_records_to_df

logger = logging.getLogger(__name__)
typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)


def find_takeout_files(base_dir: Path) -> list[Path]:
    """Find all Google Takeout Records.json files recursively in the base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise ValueError(f"Base directory is not a directory: {base_dir}")

    # Look for Records.json files (Google Takeout location history)
    takeout_files = list(base_dir.rglob("Records.json"))
    logger.info(f"Found {len(takeout_files)} Google Takeout Records.json files in {base_dir}")

    return takeout_files


def process_single_takeout_file(takeout_file: Path) -> pd.DataFrame:
    """Process a single Google Takeout Records.json file and return a DataFrame."""
    try:
        return load_google_takeout_records_to_df(takeout_file)
    except Exception as e:
        logger.error(f"Failed to process {takeout_file}: {e}")
        # Return empty DataFrame with correct schema on error
        return pd.DataFrame(
            columns=[
                C.UUID,
                C.DATA_TYPE,
                C.SOURCE_FILE,
                C.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
                C.TIMESTAMP_UTC,
                C.GPS_LATITUDE,
                C.GPS_LONGITUDE,
            ]
        )


@app.command()
def main(
    base_dir: Path = typer.Argument(
        ..., help="Directory to recursively search for Google Takeout Records.json files"
    ),
    output_parquet: Path = typer.Argument(..., help="Parquet file to write to"),
    max_workers: int = typer.Option(None, help="Maximum number of parallel workers"),
) -> None:
    """Ingest Google Takeout Records.json files into parquet format."""
    logging.basicConfig(level=logging.INFO)

    # Find all Takeout files
    takeout_files = find_takeout_files(base_dir)

    if not takeout_files:
        logger.info("No Google Takeout Records.json files found")
        return

    # Process files in parallel
    logger.info(f"Processing {len(takeout_files)} Google Takeout files...")
    dataframes = process_map(
        process_single_takeout_file,
        takeout_files,
        max_workers=max_workers,
        desc="Processing Takeout files",
        chunksize=1,
    )

    # Filter out empty DataFrames
    non_empty_dataframes = [df for df in dataframes if not df.empty]

    if not non_empty_dataframes:
        logger.info("No valid data found in Google Takeout files")
        return

    # Concatenate all DataFrames
    logger.info(f"Concatenating {len(non_empty_dataframes)} DataFrames...")
    combined_df = pd.concat(non_empty_dataframes, ignore_index=True)

    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.sort_values([C.TIMESTAMP_UTC, C.UUID], ignore_index=True)

    if combined_df[C.UUID].duplicated().any():
        logger.warning(
            "Duplicate UUIDs found in data. This should not happen if the data is clean."
        )
        dupes = combined_df[C.UUID].duplicated(keep=False)
        logger.info(f"Dupes: {combined_df[dupes][C.SOURCE_FILE].value_counts()}")
        raise ValueError("Duplicate UUIDs found in data.")

    logger.info(
        f"Total records: {len(combined_df)} (H3 columns already computed per-file in parallel)"
    )
    combined_df[C.UUID] = combined_df[C.UUID].astype('object')
    combined_df = combined_df.set_index(C.UUID, verify_integrity=True)

    logger.info(f"Writing to parquet file: {output_parquet}")
    combined_df.to_parquet(
        output_parquet,
        engine="pyarrow",
        index=True,
    )

    logger.info("Google Takeout ingestion completed successfully")


if __name__ == "__main__":
    app()