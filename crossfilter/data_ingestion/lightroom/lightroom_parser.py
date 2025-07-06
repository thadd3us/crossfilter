"""Parser for Adobe Lightroom catalog files."""

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.sqlite_utils import query_sqlite_to_dataframe

logger = logging.getLogger(__name__)


class LightroomParserConfig(BaseModel):
    """Configuration for Lightroom parser."""

    ignore_collections: Set[str] = {"quick collection"}
    include_metadata: bool = True
    include_keywords: bool = True
    include_collections: bool = True


def is_zip_file(path: Path) -> bool:
    """Check if a path is a zip file."""
    return path.suffix.lower() == ".zip"


def extract_catalog_from_zip(zip_path: Path) -> Path:
    """Extract Lightroom catalog from a zip file to a temporary directory.

    WARNING: The returned path points to a file in a temporary directory.
    The caller is responsible for cleaning up the temporary directory when done.
    """
    logger.info(f"Extracting catalog from zip file: {zip_path}")

    # Create a temporary directory for extraction
    temp_dir = Path(tempfile.mkdtemp(prefix="lightroom_catalog_"))

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the .lrcat file
        lrcat_files = list(temp_dir.glob("*.lrcat"))
        if not lrcat_files:
            raise ValueError(f"No .lrcat file found in {zip_path}")

        if len(lrcat_files) > 1:
            logger.warning(
                f"Multiple .lrcat files found in {zip_path}, using the first one"
            )

        catalog_path = lrcat_files[0]
        logger.info(f"Extracted catalog to: {catalog_path}")
        return catalog_path

    except Exception as e:
        # Clean up temp directory if extraction fails
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def determine_file_type(file_format: str) -> DataType:
    """Determine if a file is a photo or video based on Lightroom's fileFormat column."""
    if not file_format:
        return DataType.PHOTO  # Default to photo

    # Lightroom's fileFormat values for video files
    video_formats = {
        "MP4",
        "MOV",
        "AVI",
        "MKV",
        "WMV",
        "FLV",
        "WEBM",
        "M4V",
        "3GP",
        "3G2",
        "MTS",
        "M2TS",
        "TS",
        "MXF",
        "DV",
        "F4V",
        "ASF",
        "RM",
        "RMVB",
        "VOB",
        "OGV",
    }

    if file_format.upper() in video_formats:
        return DataType.VIDEO
    else:
        return DataType.PHOTO


def build_file_path(
    root_path: Optional[str], path_from_root: Optional[str], filename: Optional[str]
) -> Optional[str]:
    """Build full file path from Lightroom components."""
    if not filename:
        return None

    parts = []
    if root_path:
        parts.append(root_path)
    if path_from_root:
        parts.append(path_from_root)
    parts.append(filename)

    return os.path.join(*parts)


def parse_lightroom_catalog(
    catalog_path: Path, config: LightroomParserConfig = None
) -> pd.DataFrame:
    """Parse a Lightroom catalog file and return a DataFrame with the target schema."""
    if config is None:
        config = LightroomParserConfig()

    original_path = catalog_path
    extracted_temp_dir = None

    try:
        # Handle zip files
        if is_zip_file(catalog_path):
            catalog_path = extract_catalog_from_zip(catalog_path)
            # Remember the temp directory for cleanup
            extracted_temp_dir = catalog_path.parent

        if not catalog_path.exists():
            raise FileNotFoundError(f"Lightroom catalog not found: {catalog_path}")

        logger.info(f"Parsing Lightroom catalog: {catalog_path}")

        # Main query to get image data with basic joins
        # Start with a simpler query to ensure compatibility
        main_query = """
        SELECT DISTINCT
            -- Core Adobe_images fields
            Adobe_images.id_global,
            Adobe_images.rootFile,
            Adobe_images.captureTime,
            Adobe_images.rating,
            Adobe_images.colorLabels,
            Adobe_images.fileFormat,
            Adobe_images.originalCaptureTime,
            Adobe_images.pick,
            Adobe_images.touchTime,
            
            -- File information
            AgLibraryFile.idx_filename,
            AgLibraryFile.importHash,
            AgLibraryFile.baseName,
            AgLibraryFile.extension,
            
            -- Folder information 
            AgLibraryFolder.pathFromRoot,
            AgLibraryRootFolder.absolutePath,
            
            -- GPS and EXIF metadata (only fields that commonly exist)
            AgHarvestedExifMetadata.gpsLatitude,
            AgHarvestedExifMetadata.gpsLongitude,
            AgHarvestedExifMetadata.focalLength,
            AgHarvestedExifMetadata.aperture,
            AgHarvestedExifMetadata.flashFired,
            
            -- IPTC metadata (basic fields)
            AgLibraryIPTC.caption
            
        FROM Adobe_images
        LEFT JOIN AgLibraryFile ON AgLibraryFile.id_local = Adobe_images.rootFile
        LEFT JOIN AgLibraryFolder ON AgLibraryFolder.id_local = AgLibraryFile.folder
        LEFT JOIN AgLibraryRootFolder ON AgLibraryRootFolder.id_local = AgLibraryFolder.rootFolder
        LEFT JOIN AgHarvestedExifMetadata ON AgHarvestedExifMetadata.image = Adobe_images.id_local
        LEFT JOIN AgLibraryIPTC ON AgLibraryIPTC.image = Adobe_images.id_local
        """

        # Query the main data
        df = query_sqlite_to_dataframe(catalog_path, main_query)

        if df.empty:
            logger.warning("No images found in Lightroom catalog")
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

        logger.info(f"Found {len(df)} images in Lightroom catalog")

        # Map to target schema columns
        result_df = pd.DataFrame()

        # Required schema columns
        result_df[SchemaColumns.UUID_STRING] = df["id_global"].astype(str)
        result_df[SchemaColumns.DATA_TYPE] = df["fileFormat"].apply(determine_file_type)
        result_df[SchemaColumns.NAME] = df["idx_filename"]  # Use filename as name
        result_df[SchemaColumns.CAPTION] = df["caption"]

        # Build source file path
        result_df[SchemaColumns.SOURCE_FILE] = df.apply(
            lambda row: build_file_path(
                row.get("absolutePath"),
                row.get("pathFromRoot"),
                row.get("idx_filename"),
            ),
            axis=1,
        )

        # Handle timestamps
        result_df[SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE] = df[
            "captureTime"
        ]  # THAD: Can we actually get tz-aware timestamps?
        result_df[SchemaColumns.TIMESTAMP_UTC] = df["captureTime"]
        if result_df[SchemaColumns.TIMESTAMP_UTC].isna().any():
            logger.warning(
                f"Found {result_df[SchemaColumns.TIMESTAMP_UTC].isna().sum()} missing timestamps"
            )

        # GPS coordinates
        result_df[SchemaColumns.GPS_LATITUDE] = pd.to_numeric(
            df["gpsLatitude"], errors="coerce"
        )
        result_df[SchemaColumns.GPS_LONGITUDE] = pd.to_numeric(
            df["gpsLongitude"], errors="coerce"
        )

        # Rating (0-5)
        result_df[SchemaColumns.RATING_0_TO_5] = (
            pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype("Int64")
        )

        # File size (not available in simplified query, set to None)
        result_df[SchemaColumns.SIZE_IN_BYTES] = None

        # Add useful metadata columns as extras (not part of required schema)
        if config.include_metadata:
            # EXIF data (available from our simplified query)
            result_df["focal_length"] = pd.to_numeric(
                df["focalLength"], errors="coerce"
            )
            result_df["aperture"] = pd.to_numeric(df["aperture"], errors="coerce")
            result_df["flash_fired"] = df["flashFired"]

            # File metadata
            result_df["import_hash"] = df["importHash"]
            result_df["file_format"] = df["fileFormat"]
            result_df["base_name"] = df["baseName"]
            result_df["extension"] = df["extension"]

            # Lightroom-specific fields
            result_df["color_labels"] = df["colorLabels"]
            result_df["pick"] = df["pick"]
            result_df["touch_time"] = df["touchTime"]
            result_df["original_capture_time"] = df["originalCaptureTime"]

        # Add keywords if requested
        if config.include_keywords:
            keywords_df = _get_keywords_for_images(
                catalog_path, result_df[SchemaColumns.UUID_STRING].tolist()
            )
            result_df = result_df.merge(
                keywords_df, on=SchemaColumns.UUID_STRING, how="left"
            )
            # Convert keywords list to JSON string for SQLite storage
            if "keywords" in result_df.columns:
                result_df["keywords"] = result_df["keywords"].apply(
                    lambda x: str(x) if x is not None else None
                )

        # Add collections if requested
        if config.include_collections:
            collections_df = _get_collections_for_images(
                catalog_path,
                result_df[SchemaColumns.UUID_STRING].tolist(),
                config.ignore_collections,
            )
            result_df = result_df.merge(
                collections_df, on=SchemaColumns.UUID_STRING, how="left"
            )
            # Convert collections list to JSON string for SQLite storage
            if "collections" in result_df.columns:
                result_df["collections"] = result_df["collections"].apply(
                    lambda x: str(x) if x is not None else None
                )

        logger.info(f"Parsed {len(result_df)} records from Lightroom catalog")

        return result_df

    finally:
        # Clean up temporary directory if we extracted from zip
        if extracted_temp_dir:
            try:
                import shutil

                shutil.rmtree(extracted_temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory {extracted_temp_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary directory {extracted_temp_dir}: {e}"
                )


def _get_keywords_for_images(catalog_path: Path, image_ids: List[str]) -> pd.DataFrame:
    """Get keywords for the specified image IDs."""
    if not image_ids:
        return pd.DataFrame(columns=[SchemaColumns.UUID_STRING, "keywords"])

    # Convert image_ids to a format suitable for SQL IN clause
    ids_str = ",".join(f"'{id_}'" for id_ in image_ids)

    keywords_query = f"""
    SELECT 
        Adobe_images.id_global,
        AgLibraryKeyword.name as keyword_name
    FROM AgLibraryKeywordImage
    LEFT JOIN Adobe_images ON Adobe_images.id_local = AgLibraryKeywordImage.image
    LEFT JOIN AgLibraryKeyword ON AgLibraryKeyword.id_local = AgLibraryKeywordImage.tag
    WHERE Adobe_images.id_global IN ({ids_str})
    AND AgLibraryKeyword.name IS NOT NULL
    """

    try:
        keywords_df = query_sqlite_to_dataframe(catalog_path, keywords_query)
        if not keywords_df.empty:
            # Group keywords by image ID
            grouped = (
                keywords_df.groupby("id_global")["keyword_name"]
                .apply(lambda x: sorted(x.dropna().tolist()))
                .reset_index()
            )
            grouped.columns = [SchemaColumns.UUID_STRING, "keywords"]
            grouped[SchemaColumns.UUID_STRING] = grouped[
                SchemaColumns.UUID_STRING
            ].astype(str)
            return grouped
    except Exception as e:
        logger.warning(f"Failed to get keywords: {e}")

    return pd.DataFrame(columns=[SchemaColumns.UUID_STRING, "keywords"])


def _get_collections_for_images(
    catalog_path: Path, image_ids: List[str], ignore_collections: Set[str]
) -> pd.DataFrame:
    """Get collections for the specified image IDs."""
    if not image_ids:
        return pd.DataFrame(columns=[SchemaColumns.UUID_STRING, "collections"])

    # Convert image_ids to a format suitable for SQL IN clause
    ids_str = ",".join(f"'{id_}'" for id_ in image_ids)

    # Convert ignore_collections to SQL format (case-insensitive)
    ignore_str = ",".join(f"'{col.lower()}'" for col in ignore_collections)
    ignore_clause = (
        f"AND LOWER(AgLibraryCollection.name) NOT IN ({ignore_str})"
        if ignore_collections
        else ""
    )

    collections_query = f"""
    SELECT 
        Adobe_images.id_global,
        AgLibraryCollection.name as collection_name
    FROM AgLibraryCollectionImage
    LEFT JOIN Adobe_images ON Adobe_images.id_local = AgLibraryCollectionImage.image
    LEFT JOIN AgLibraryCollection ON AgLibraryCollection.id_local = AgLibraryCollectionImage.collection
    WHERE Adobe_images.id_global IN ({ids_str})
    AND AgLibraryCollection.name IS NOT NULL
    {ignore_clause}
    """

    try:
        collections_df = query_sqlite_to_dataframe(catalog_path, collections_query)
        if not collections_df.empty:
            # Group collections by image ID
            grouped = (
                collections_df.groupby("id_global")["collection_name"]
                .apply(lambda x: sorted(x.dropna().tolist()))
                .reset_index()
            )
            grouped.columns = [SchemaColumns.UUID_STRING, "collections"]
            grouped[SchemaColumns.UUID_STRING] = grouped[
                SchemaColumns.UUID_STRING
            ].astype(str)
            return grouped
    except Exception as e:
        logger.warning(f"Failed to get collections: {e}")

    return pd.DataFrame(columns=[SchemaColumns.UUID_STRING, "collections"])


def load_lightroom_catalog_to_df(
    catalog_path: Path, config: LightroomParserConfig = None
) -> pd.DataFrame:
    """Load a Lightroom catalog and return a DataFrame with the target schema.

    This is the main entry point for loading Lightroom catalogs.
    """
    try:
        return parse_lightroom_catalog(catalog_path, config)
    except Exception as e:
        logger.error(f"Failed to load Lightroom catalog {catalog_path}: {e}")
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
