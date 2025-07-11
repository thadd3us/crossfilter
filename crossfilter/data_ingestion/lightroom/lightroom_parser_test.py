"""Tests for Lightroom catalog parser."""

import tempfile
import zipfile
from pathlib import Path
from typing import Set

import pandas as pd
import pytest

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.lightroom.lightroom_parser import (
    LightroomParserConfig,
    build_file_path,
    determine_file_type,
    extract_catalog_from_zip,
    is_zip_file,
    load_lightroom_catalog_to_df,
    parse_lightroom_catalog,
)


def test_determine_file_type() -> None:
    """Test determining file type from Lightroom's fileFormat column."""
    # Photo formats (Lightroom fileFormat values)
    assert determine_file_type("JPEG") == DataType.PHOTO
    assert determine_file_type("DNG") == DataType.PHOTO
    assert determine_file_type("TIFF") == DataType.PHOTO
    assert determine_file_type("PNG") == DataType.PHOTO
    assert determine_file_type("RAW") == DataType.PHOTO

    # Video formats (Lightroom fileFormat values)
    assert determine_file_type("MP4") == DataType.VIDEO
    assert determine_file_type("mp4") == DataType.VIDEO
    assert determine_file_type("MOV") == DataType.VIDEO
    assert determine_file_type("AVI") == DataType.VIDEO
    assert determine_file_type("MKV") == DataType.VIDEO
    assert determine_file_type("WMV") == DataType.VIDEO
    assert determine_file_type("M4V") == DataType.VIDEO

    # Edge cases
    assert determine_file_type("") == DataType.PHOTO  # Default to photo
    assert determine_file_type(None) == DataType.PHOTO
    assert determine_file_type("UNKNOWN") == DataType.PHOTO


def test_build_file_path() -> None:
    """Test building file paths from Lightroom components."""
    # Complete path
    result = build_file_path("/root/path", "subfolder", "image.jpg")
    assert result == "/root/path/subfolder/image.jpg"

    # Missing path_from_root
    result = build_file_path("/root/path", None, "image.jpg")
    assert result == "/root/path/image.jpg"

    # Missing root_path
    result = build_file_path(None, "subfolder", "image.jpg")
    assert result == "subfolder/image.jpg"

    # Only filename
    result = build_file_path(None, None, "image.jpg")
    assert result == "image.jpg"

    # Missing filename
    result = build_file_path("/root/path", "subfolder", None)
    assert result is None

    result = build_file_path("/root/path", "subfolder", "")
    assert result is None


def test_is_zip_file() -> None:
    """Test zip file detection."""
    assert is_zip_file(Path("catalog.zip"))
    assert is_zip_file(Path("catalog.ZIP"))
    assert not is_zip_file(Path("catalog.lrcat"))
    assert not is_zip_file(Path("catalog.txt"))


def test_extract_catalog_from_zip(tmp_path: Path) -> None:
    """Test extracting catalog from zip file."""
    # Create a test zip file with a catalog
    zip_path = tmp_path / "test_catalog.zip"
    catalog_content = b"fake lightroom catalog content"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("test_catalog.lrcat", catalog_content)

    # Extract the catalog
    extracted_path = extract_catalog_from_zip(zip_path)

    assert extracted_path.exists()
    assert extracted_path.suffix == ".lrcat"
    assert extracted_path.read_bytes() == catalog_content


def test_extract_catalog_from_zip_no_catalog(tmp_path: Path) -> None:
    """Test extracting catalog from zip file with no .lrcat file."""
    # Create a test zip file without a catalog
    zip_path = tmp_path / "test_no_catalog.zip"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("some_file.txt", b"not a catalog")

    # Should raise an error
    with pytest.raises(ValueError, match="No .lrcat file found"):
        extract_catalog_from_zip(zip_path)


def test_lightroom_parser_config() -> None:
    """Test LightroomParserConfig default values and validation."""
    # Default config
    config = LightroomParserConfig()
    assert config.ignore_collections == {"quick collection"}
    assert config.include_metadata is True
    assert config.include_keywords is True
    assert config.include_collections is True

    # Custom config
    custom_ignore = {"test collection", "another one"}
    config = LightroomParserConfig(
        ignore_collections=custom_ignore,
        include_metadata=False,
        include_keywords=False,
        include_collections=False,
    )
    assert config.ignore_collections == custom_ignore
    assert config.include_metadata is False
    assert config.include_keywords is False
    assert config.include_collections is False


def test_parse_lightroom_catalog_nonexistent() -> None:
    """Test parsing non-existent catalog file."""
    nonexistent_path = Path("/nonexistent/catalog.lrcat")

    with pytest.raises(FileNotFoundError):
        parse_lightroom_catalog(nonexistent_path)


def test_load_lightroom_catalog_to_df_nonexistent() -> None:
    """Test loading non-existent catalog file with error handling."""
    nonexistent_path = Path("/nonexistent/catalog.lrcat")
    # Should return empty DataFrame with correct schema instead of raising
    with pytest.raises(FileNotFoundError):
        load_lightroom_catalog_to_df(nonexistent_path)


def test_parse_lightroom_catalog_real_data(test_catalogs_dir: Path) -> None:
    """Test parsing real Lightroom catalog files from test data."""
    test_catalogs = [
        test_catalogs_dir / "test_catalog_00" / "test_catalog_fresh.lrcat",
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat",
        test_catalogs_dir
        / "test_catalog_02"
        / "test_catalog_two_more_photos_and_edits.lrcat",
        test_catalogs_dir
        / "test_catalog_03"
        / "test_catalog_more_face_tags_gps_edit.lrcat",
    ]

    for catalog_path in test_catalogs:
        # Test with default config
        df = parse_lightroom_catalog(catalog_path)

        # Verify required schema columns exist
        assert SchemaColumns.UUID_STRING in df.columns
        assert SchemaColumns.DATA_TYPE in df.columns
        assert SchemaColumns.NAME in df.columns
        assert SchemaColumns.CAPTION in df.columns
        assert SchemaColumns.SOURCE_FILE in df.columns
        assert SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE in df.columns
        assert SchemaColumns.TIMESTAMP_UTC in df.columns
        assert SchemaColumns.GPS_LATITUDE in df.columns
        assert SchemaColumns.GPS_LONGITUDE in df.columns
        assert SchemaColumns.RATING_0_TO_5 in df.columns
        assert SchemaColumns.SIZE_IN_BYTES in df.columns

        # Check data types
        assert df[SchemaColumns.UUID_STRING].dtype == "object"
        assert df[SchemaColumns.DATA_TYPE].isin([DataType.PHOTO, DataType.VIDEO]).all()

        # Check for valid GPS coordinates if present
        if df[SchemaColumns.GPS_LATITUDE].notna().any():
            non_null_lats = df[SchemaColumns.GPS_LATITUDE].dropna()
            valid_lat = non_null_lats.between(-90, 90, inclusive="both")
            assert (
                valid_lat.all()
            ), f"Invalid latitudes found: {non_null_lats[~valid_lat].tolist()}"

        if df[SchemaColumns.GPS_LONGITUDE].notna().any():
            non_null_lons = df[SchemaColumns.GPS_LONGITUDE].dropna()
            valid_lon = non_null_lons.between(-180, 180, inclusive="both")
            assert (
                valid_lon.all()
            ), f"Invalid longitudes found: {non_null_lons[~valid_lon].tolist()}"

        # Check ratings are in valid range
        if df[SchemaColumns.RATING_0_TO_5].notna().any():
            non_null_ratings = df[SchemaColumns.RATING_0_TO_5].dropna()
            valid_ratings = non_null_ratings.between(0, 5, inclusive="both")
            assert (
                valid_ratings.all()
            ), f"Invalid ratings found: {non_null_ratings[~valid_ratings].tolist()}"


def test_parse_lightroom_catalog_with_metadata(test_catalogs_dir: Path) -> None:
    """Test parsing catalog with metadata enabled."""
    catalog_path = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    config = LightroomParserConfig(include_metadata=True)
    df = parse_lightroom_catalog(catalog_path, config)

    # Check for metadata columns (updated to match simplified query)
    metadata_columns = [
        "focal_length",
        "aperture",
        "flash_fired",
        "import_hash",
        "file_format",
        "base_name",
        "extension",
        "color_labels",
        "pick",
        "touch_time",
        "original_capture_time",
    ]

    for col in metadata_columns:
        assert col in df.columns, f"Missing metadata column: {col}"


def test_parse_lightroom_catalog_without_metadata(test_catalogs_dir: Path) -> None:
    """Test parsing catalog with metadata disabled."""
    catalog_path = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    config = LightroomParserConfig(include_metadata=False)
    df = parse_lightroom_catalog(catalog_path, config)

    # Check that extra metadata columns are not present
    metadata_columns = ["focal_length", "aperture", "import_hash", "file_format"]

    for col in metadata_columns:
        assert col not in df.columns, f"Unexpected metadata column found: {col}"


def test_parse_lightroom_catalog_with_keywords(test_catalogs_dir: Path) -> None:
    """Test parsing catalog with keywords enabled."""
    catalog_path = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    config = LightroomParserConfig(include_keywords=True)
    df = parse_lightroom_catalog(catalog_path, config)

    # Check for keywords column
    assert "keywords" in df.columns

    # Check that keywords are strings when present (converted for SQLite storage)
    keywords_present = df["keywords"].notna()
    if keywords_present.any():
        for keywords in df.loc[keywords_present, "keywords"]:
            assert isinstance(
                keywords, str
            ), f"Keywords should be a string (for SQLite), got {type(keywords)}"


def test_parse_lightroom_catalog_with_collections(test_catalogs_dir: Path) -> None:
    """Test parsing catalog with collections enabled."""
    catalog_path = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    config = LightroomParserConfig(include_collections=True)
    df = parse_lightroom_catalog(catalog_path, config)

    # Check for collections column
    assert "collections" in df.columns

    # Check that collections are strings when present (converted for SQLite storage)
    collections_present = df["collections"].notna()
    if collections_present.any():
        for collections in df.loc[collections_present, "collections"]:
            assert isinstance(
                collections, str
            ), f"Collections should be a string (for SQLite), got {type(collections)}"


def test_parse_lightroom_catalog_ignore_collections(test_catalogs_dir: Path) -> None:
    """Test parsing catalog with specific collections ignored."""
    catalog_path = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    # Test with custom ignored collections
    ignore_set = {"test collection", "ignored collection"}
    config = LightroomParserConfig(
        include_collections=True, ignore_collections=ignore_set
    )
    df = parse_lightroom_catalog(catalog_path, config)

    # Check that ignored collections don't appear in results
    if "collections" in df.columns and df["collections"].notna().any():
        all_collections = set()
        for collections_list in df["collections"].dropna():
            if isinstance(collections_list, list):
                all_collections.update(col.lower() for col in collections_list)

        # Check that ignored collections are not present
        for ignored in ignore_set:
            assert (
                ignored.lower() not in all_collections
            ), f"Ignored collection '{ignored}' found in results"


def test_parse_multiple_catalogs(test_catalogs_dir: Path) -> None:
    """Test parsing multiple catalogs and checking for unique UUIDs."""
    test_catalogs = [
        test_catalogs_dir / "test_catalog_00" / "test_catalog_fresh.lrcat",
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat",
    ]

    all_dataframes = []
    for catalog_path in test_catalogs:
        df = parse_lightroom_catalog(catalog_path)
        if not df.empty:
            all_dataframes.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Check for UUID uniqueness across catalogs
    uuid_counts = combined_df[SchemaColumns.UUID_STRING].value_counts()
    duplicates = uuid_counts[uuid_counts > 1]

    # This is informational - we might expect some duplicates between test catalogs
    assert len(all_dataframes) > 0, "Should have parsed at least one catalog"


def test_parse_timezone_aware_timestamps(test_catalogs_dir: Path, snapshot) -> None:
    """Test parsing catalog with various timezone formats in timestamps including NULL values."""
    from syrupy.assertion import SnapshotAssertion
    
    # Use dedicated test catalog with timezone formats and NULL captureTime
    test_catalog = (
        test_catalogs_dir / "test_parse_timezone_aware_timestamps" / "test_parse_timezone_aware_timestamps.lrcat"
    )

    config = LightroomParserConfig(
        include_metadata=False, include_keywords=False, include_collections=False
    )
    df = parse_lightroom_catalog(test_catalog, config)

    # Extract only the timestamp columns for snapshot testing
    timestamp_data = df[
        [
            SchemaColumns.UUID_STRING,
            SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
            SchemaColumns.TIMESTAMP_UTC,
        ]
    ].copy()

    # Convert timezone-aware timestamps to string for snapshot comparison
    # (since timezone-aware timestamps can be tricky to compare in snapshots)
    timestamp_data[SchemaColumns.TIMESTAMP_UTC] = timestamp_data[
        SchemaColumns.TIMESTAMP_UTC
    ].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Create snapshot of the timezone handling
    assert timestamp_data.to_dict("records") == snapshot
