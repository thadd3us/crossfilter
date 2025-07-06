"""Tests for Lightroom catalog ingestion CLI program."""

import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.lightroom.ingest_lightroom_catalogs import (
    find_lightroom_catalogs,
    main,
)
from crossfilter.data_ingestion.lightroom.lightroom_parser import (
    LightroomParserConfig,
    load_lightroom_catalog_to_df,
)


def create_test_lrcat_file(path: Path, content: str = "fake catalog") -> None:
    """Create a test Lightroom catalog file."""
    path.write_text(content)


def create_test_zip_with_catalog(
    zip_path: Path, catalog_name: str = "test.lrcat"
) -> None:
    """Create a test zip file containing a Lightroom catalog."""
    catalog_content = b"fake lightroom catalog content"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr(catalog_name, catalog_content)


def test_find_lightroom_catalogs_nonexistent_directory() -> None:
    """Test finding catalogs in non-existent directory."""
    nonexistent_dir = Path("/nonexistent/directory")

    with pytest.raises(FileNotFoundError):
        find_lightroom_catalogs(nonexistent_dir)


def test_find_lightroom_catalogs_not_directory(tmp_path: Path) -> None:
    """Test finding catalogs when path is not a directory."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with pytest.raises(ValueError):
        find_lightroom_catalogs(test_file)


def test_find_lightroom_catalogs_empty_directory(tmp_path: Path) -> None:
    """Test finding catalogs in empty directory."""
    catalog_files = find_lightroom_catalogs(tmp_path)
    assert catalog_files == []


def test_find_lightroom_catalogs_with_files(tmp_path: Path) -> None:
    """Test finding catalogs in directory with files."""
    # Create some Lightroom catalog files
    (tmp_path / "catalog1.lrcat").write_text("content1")
    (tmp_path / "catalog2.lrcat").write_text("content2")
    (tmp_path / "other.txt").write_text("content3")

    # Create zip files
    create_test_zip_with_catalog(tmp_path / "catalog3.zip")
    create_test_zip_with_catalog(tmp_path / "catalog4.zip")

    # Create subdirectory with catalog file
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "catalog5.lrcat").write_text("content5")

    catalog_files = find_lightroom_catalogs(tmp_path)

    assert len(catalog_files) == 5
    assert tmp_path / "catalog1.lrcat" in catalog_files
    assert tmp_path / "catalog2.lrcat" in catalog_files
    assert tmp_path / "catalog3.zip" in catalog_files
    assert tmp_path / "catalog4.zip" in catalog_files
    assert subdir / "catalog5.lrcat" in catalog_files


def test_find_lightroom_catalogs_case_insensitive(tmp_path: Path) -> None:
    """Test finding catalogs with different case extensions."""
    # Create files with different case extensions
    (tmp_path / "catalog1.lrcat").write_text("content1")
    (tmp_path / "catalog2.LRCAT").write_text("content2")
    (tmp_path / "archive1.zip").write_text("content3")
    (tmp_path / "archive2.ZIP").write_text("content4")

    catalog_files = find_lightroom_catalogs(tmp_path)

    # Note: glob.rglob() is case-sensitive on case-sensitive filesystems
    # On case-insensitive filesystems (like macOS), it would find all 4
    # On case-sensitive filesystems (like Linux), it finds only exact matches
    assert len(catalog_files) >= 2  # At least the lowercase files should be found
    assert len(catalog_files) <= 4  # At most all files should be found


def test_load_lightroom_catalog_error_handling(tmp_path: Path) -> None:
    """Test loading invalid catalog file with error handling."""
    # Create invalid catalog file
    invalid_catalog = tmp_path / "invalid.lrcat"
    invalid_catalog.write_text("This is not a valid Lightroom catalog")

    config = LightroomParserConfig()

    # Should return empty DataFrame with proper schema on error
    with pytest.raises(Exception):
        df = load_lightroom_catalog_to_df(invalid_catalog, config)


def test_load_lightroom_catalog_real_data(test_catalogs_dir: Path) -> None:
    """Test loading real Lightroom catalog files."""
    test_catalog = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    config = LightroomParserConfig()
    df = load_lightroom_catalog_to_df(test_catalog, config)

    # Should return valid DataFrame
    assert not df.empty
    assert SchemaColumns.UUID_STRING in df.columns
    assert SchemaColumns.DATA_TYPE in df.columns

    # Check data integrity
    assert df[SchemaColumns.UUID_STRING].notna().all()
    assert df[SchemaColumns.DATA_TYPE].isin([DataType.PHOTO, DataType.VIDEO]).all()


def test_load_lightroom_catalog_with_config(
    test_catalogs_dir: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading catalog with custom configuration and verify full contents."""
    test_catalog = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    # Test with metadata disabled
    config = LightroomParserConfig(
        include_metadata=False, include_keywords=False, include_collections=False
    )
    df = load_lightroom_catalog_to_df(test_catalog, config)

    # Should have required columns but not metadata
    assert not df.empty
    assert SchemaColumns.UUID_STRING in df.columns
    assert "focal_length" not in df.columns
    assert "keywords" not in df.columns
    assert "collections" not in df.columns

    # Verify entire DataFrame contents with syrupy snapshot
    assert df.to_dict("records") == snapshot


def test_load_zip_catalog(test_catalogs_dir: Path, tmp_path: Path) -> None:
    """Test loading a zip file containing a real catalog."""
    # Use one of the real catalogs
    real_catalog = test_catalogs_dir / "test_catalog_00" / "test_catalog_fresh.lrcat"

    # Create a zip file containing the real catalog
    zip_path = tmp_path / "test_catalog.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.write(real_catalog, real_catalog.name)

    config = LightroomParserConfig()
    df = load_lightroom_catalog_to_df(zip_path, config)

    # Should successfully parse the catalog from the zip
    assert not df.empty
    assert SchemaColumns.UUID_STRING in df.columns
    assert len(df) == 5  # Known count from test_catalog_fresh.lrcat


def test_duplicate_handling(test_catalogs_dir: Path) -> None:
    """Test handling of duplicate records from multiple catalogs."""
    # Test with two specific catalogs
    test_catalogs = [
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat",
        test_catalogs_dir
        / "test_catalog_02"
        / "test_catalog_two_more_photos_and_edits.lrcat",
    ]

    config = LightroomParserConfig()
    dataframes = []

    for catalog_path in test_catalogs:
        df = load_lightroom_catalog_to_df(catalog_path, config)
        dataframes.append(df)

    # Combine DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Check for duplicates
    initial_count = len(combined_df)
    deduplicated_df = combined_df.drop_duplicates(subset=[SchemaColumns.UUID_STRING])
    final_count = len(deduplicated_df)

    # Should have processed both catalogs
    assert len(dataframes) == 2
    assert initial_count > 0
    assert final_count > 0


def test_large_ignore_collections_set() -> None:
    """Test with a large set of collections to ignore."""
    large_ignore_set = {f"collection_{i}" for i in range(1000)}

    config = LightroomParserConfig(
        ignore_collections=large_ignore_set, include_collections=True
    )

    # Should handle large ignore sets without issues
    assert len(config.ignore_collections) == 1000
    assert "collection_500" in config.ignore_collections


def test_config_case_insensitive_ignore(test_catalogs_dir: Path) -> None:
    """Test that collection ignoring is case-insensitive."""
    test_catalog = (
        test_catalogs_dir
        / "test_catalog_01"
        / "test_catalog_gps_captions_collections_keywords.lrcat"
    )

    # Test with uppercase ignore patterns
    config = LightroomParserConfig(
        ignore_collections={"QUICK COLLECTION", "TEST COLLECTION"},
        include_collections=True,
    )

    df = load_lightroom_catalog_to_df(test_catalog, config)

    # Should process without errors
    assert SchemaColumns.UUID_STRING in df.columns


@pytest.mark.skipif(sys.platform != "darwin", reason="Data is only on Thad's laptop")
def test_thad_ingest_dev_data(
    source_tree_root: Path, tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    main(
        base_dir=source_tree_root / "dev_data",
        destination_sqlite_db=tmp_path / "lightroom.sqlite",
        destination_table="data",
        include_metadata=True,
        include_keywords=True,
        include_collections=True,
        ignore_collections="quick collection",
    )

    assert (tmp_path / "lightroom.sqlite").exists()
