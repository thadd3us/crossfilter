"""CLI program for ingesting Lightroom catalog files into SQLite database."""

import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import dogpile.cache
import msgpack_numpy as msgpack
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from crossfilter.core.bucketing import (
    add_semantic_embedding_umap_h3_bucket_columns,
    add_geo_h3_bucket_columns,
)
from crossfilter.core.schema import SchemaColumns
from crossfilter.data_ingestion.lightroom.lightroom_parser import (
    LightroomParserConfig,
    load_lightroom_catalog_to_df,
)
from crossfilter.data_ingestion.sqlite_utils import upsert_dataframe_to_sqlite

logger = logging.getLogger(__name__)

# Configure dogpile.cache
cache_region = dogpile.cache.make_region().configure(
    "dogpile.cache.dbm", arguments={"filename": "/tmp/crossfilter_dogpile_cache.dbm"}
)


# @cache_region.cache_on_arguments()
def load_clip_embeddings_from_sqlite(sqlite_db_path: Path) -> pd.DataFrame:
    """Load CLIP embeddings from SQLite database and return as DataFrame.

    Args:
        sqlite_db_path: Path to SQLite database with CLIP embeddings

    Returns:
        DataFrame with UUID_STRING and embedding columns
    """
    if not sqlite_db_path.exists():
        raise FileNotFoundError(f"CLIP embeddings database not found: {sqlite_db_path}")

    logger.info(f"Loading CLIP embeddings from {sqlite_db_path}")

    with sqlite3.connect(sqlite_db_path) as conn:
        # Load embeddings table
        df = pd.read_sql(
            """SELECT * FROM embeddings WHERE type_index = "CLIP_HF_EMBEDDINGS" """,
            conn,
        )

    # Rename uuid_index to UUID_STRING to match schema
    df = df.rename(columns={"uuid_index": SchemaColumns.UUID_STRING})

    # Unpack msgpack embeddings and convert to numpy arrays
    def unpack_embedding(msgpack_data):
        unpacked = msgpack.unpackb(msgpack_data)
        return unpacked

    df["embedding"] = df["embedding_msgpack"].map(unpack_embedding)

    # Drop the msgpack column as we now have the unpacked version
    df = df.drop(columns=["embedding_msgpack", "type_index"])

    logger.info(f"Loaded {len(df)} CLIP embeddings")
    return df


# @cache_region.cache_on_arguments()
def compute_umap_projection(embeddings_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Normalize embeddings and compute 2D UMAP projection with cosine metric.

    This function normalizes CLIP embeddings to unit length and computes a 2D UMAP projection
    using the cosine metric, which is appropriate for normalized high-dimensional embeddings.
    The resulting 2D coordinates are treated as spherical coordinates (latitude/longitude).

    References:
        https://umap-learn.readthedocs.io/en/latest/embedding_space.html

    Args:
        embeddings_df: DataFrame with UUID_STRING and embedding columns
        output_file: Optional path to save the UMAP transformation object

    Returns:
        Tuple of (DataFrame with UMAP coordinates, UMAP transformation object)
    """
    # embeddings_df = embeddings_df.head(5000)  # TODO: Remove this.

    import umap

    logger.info(f"Computing UMAP projection for {len(embeddings_df)} embeddings")

    # Check minimum number of points for UMAP
    if len(embeddings_df) < 4:
        logger.warning(
            f"UMAP requires at least 4 points, got {len(embeddings_df)}. Using fallback to simple projection."
        )
        # For very small datasets, just use the first two dimensions as a fallback
        embeddings_list = embeddings_df["embedding"].tolist()
        embeddings_array = np.array(embeddings_list)

        # Use first two dimensions or pad with zeros if needed
        if embeddings_array.shape[1] >= 2:
            umap_embedding = embeddings_array[:, :2]
        else:
            umap_embedding = np.column_stack(
                [
                    (
                        embeddings_array[:, 0]
                        if embeddings_array.shape[1] >= 1
                        else np.zeros(len(embeddings_df))
                    ),
                    np.zeros(len(embeddings_df)),
                ]
            )

        umap_transformer = None  # No transformer for fallback case
    else:
        # Extract embeddings and convert to numpy array
        embeddings_list = embeddings_df["embedding"].tolist()
        embeddings_array = np.array(embeddings_list)

        # Normalize embeddings to unit length
        embeddings_normalized = embeddings_array / np.linalg.norm(
            embeddings_array, axis=1, keepdims=True
        )

        # Compute UMAP projection with cosine metric for high-dimensional embeddings
        # The output will be treated as spherical coordinates, hence the "haversine" in the column names
        # https://umap-learn.readthedocs.io/en/latest/embedding_space.html
        umap_transformer = umap.UMAP(
            n_components=2,
            metric="euclidean",
            random_state=42,  # For reproducibility
            verbose=True,
            output_metric="haversine",
        )

        umap_embedding = umap_transformer.fit_transform(embeddings_normalized)

    # Create result DataFrame
    result_df = embeddings_df[[SchemaColumns.UUID_STRING]].copy()
    # Latitude should be -90 to 90, longitude should be -180 to 180.
    result_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE] = umap_embedding[:, 1]
    result_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE] = umap_embedding[:, 0]

    logger.info(
        f"UMAP projection completed successfully, {result_df.shape=}, {result_df.columns=}"
    )
    return result_df, umap_transformer


def find_lightroom_catalogs(base_dir: Path) -> list[Path]:
    """Find all Lightroom catalog files recursively in the base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise ValueError(f"Base directory is not a directory: {base_dir}")

    # Find both .lrcat files and .zip files (which might contain catalogs)
    lrcat_files = list(base_dir.rglob("*.lrcat"))
    zip_files = list(base_dir.rglob("*.zip"))

    all_catalog_files = lrcat_files + zip_files
    logger.info(
        f"Found {len(lrcat_files)} .lrcat files and {len(zip_files)} .zip files in {base_dir}"
    )

    return all_catalog_files


cli = typer.Typer(
    help="Ingest Lightroom catalog files into SQLite database",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


@cli.command("ingest")
def ingest(
    base_dir: Path = typer.Argument(
        ...,
        help="Directory to recursively search for Lightroom catalog files (.lrcat and .zip)",
    ),
    destination_sqlite_db: Path = typer.Argument(
        ..., help="SQLite database file to upsert into"
    ),
    destination_table: str = typer.Option("data", help="Database table name"),
    include_metadata: bool = typer.Option(
        True, help="Include camera and EXIF metadata"
    ),
    include_keywords: bool = typer.Option(True, help="Include keyword tags"),
    include_collections: bool = typer.Option(
        True, help="Include collection/album information"
    ),
    ignore_collections: str = typer.Option(
        "quick collection",
        help="Comma-separated list of collection names to ignore (case-insensitive)",
    ),
    sqlite_db_with_clip_embeddings: Optional[Path] = typer.Option(
        None,
        help="SQLite database containing CLIP embeddings associated with each UUID",
    ),
    output_umap_transformation_file: Optional[Path] = typer.Option(
        None, help="File to save UMAP transformation object for later use"
    ),
) -> None:
    """Ingest Lightroom catalog files into SQLite database."""
    logging.basicConfig(level=logging.INFO)

    # Parse ignore_collections
    ignore_set = set()
    if ignore_collections:
        ignore_set = {
            col.strip().lower() for col in ignore_collections.split(",") if col.strip()
        }

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
        logger.error("No Lightroom catalog files found")
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
        duplicated_uuids = combined_df[
            combined_df[SchemaColumns.UUID_STRING].duplicated(keep=False)
        ]
        logger.error(
            f"Duplicated UUIDs: {duplicated_uuids[SchemaColumns.UUID_STRING].unique()}"
        )
        # Remove duplicates, keeping the first occurrence
        combined_df = combined_df.drop_duplicates(
            subset=[SchemaColumns.UUID_STRING], keep="first"
        )

    logger.info(f"Total records: {len(combined_df)}")

    # Add H3 columns at ingestion time for faster runtime performance
    if (
        SchemaColumns.GPS_LATITUDE in combined_df.columns
        and SchemaColumns.GPS_LONGITUDE in combined_df.columns
    ):
        logger.info("Adding H3 spatial index columns during ingestion...")
        add_geo_h3_bucket_columns(combined_df)
        logger.info(f"Added H3 columns to {len(combined_df)} rows")

    # Process CLIP embeddings if provided
    if sqlite_db_with_clip_embeddings:
        logger.info("Processing CLIP embeddings...")

        # Load CLIP embeddings
        embeddings_df = load_clip_embeddings_from_sqlite(sqlite_db_with_clip_embeddings)
        assert embeddings_df[SchemaColumns.UUID_STRING].notna().all()
        assert not embeddings_df[SchemaColumns.UUID_STRING].duplicated().any()

        # Compute UMAP projection
        umap_coords_df, umap_transformer = compute_umap_projection(embeddings_df)

        if umap_transformer is not None:
            logger.info(
                f"Saving UMAP transformation to {output_umap_transformation_file}"
            )
            with open(str(output_umap_transformation_file), "wb") as f:
                pickle.dump(umap_transformer, f)
        logger.info(
            f"UMAP transformation saved successfully, {len(umap_coords_df)} rows"
        )

        # Merge UMAP coordinates with main DataFrame (LEFT JOIN)
        logger.info("Merging CLIP UMAP coordinates with main DataFrame...")
        combined_df = pd.merge(
            combined_df, umap_coords_df, on=SchemaColumns.UUID_STRING, how="left"
        )
        logger.info(
            f"Combined DataFrame with {len(combined_df)=} rows and {len(umap_coords_df)=} rows with CLIP embeddings"
        )

        # Add H3 columns for CLIP UMAP coordinates
        if (
            SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE in combined_df.columns
            and SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE in combined_df.columns
        ):
            logger.info("Adding H3 spatial index columns for CLIP UMAP coordinates...")
            add_semantic_embedding_umap_h3_bucket_columns(combined_df)
            logger.info(f"Added CLIP UMAP H3 columns to {len(combined_df)} rows")

        # Log statistics
        rows_with_clip_embeddings = (
            combined_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE].notna().sum()
        )
        logger.info(
            f"Successfully processed {rows_with_clip_embeddings} rows with CLIP embeddings"
        )

    # Upsert to database
    upsert_dataframe_to_sqlite(combined_df, destination_sqlite_db, destination_table)

    logger.info("Lightroom catalog ingestion completed successfully")


# https://github.com/fastapi/typer/issues/341
typer.main.get_command_name = lambda name: name


def main() -> None:
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()
