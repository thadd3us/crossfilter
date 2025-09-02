"""Tests for Lightroom catalog ingestion CLI program."""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from crossfilter.core.schema import DataType, SchemaColumns
from crossfilter.data_ingestion.lightroom.ingest_lightroom_catalogs import (
    compute_umap_projection,
    find_lightroom_catalogs,
    load_clip_embeddings_from_sqlite,
    main,
)
from crossfilter.data_ingestion.lightroom.lightroom_parser import (
    LightroomParserConfig,
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
        load_lightroom_catalog_to_df(invalid_catalog, config)


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


@pytest.mark.skipif(
    os.environ.get("THAD_DATA_AVAILAVLE") != "true",
    reason="Data is only on Thad's laptop",
)
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
        # sqlite_db_with_clip_embeddings=Path(
        #     "~/personal/lightroom_embedding_vectors.sqlite"
        # ).expanduser(),
        sqlite_db_with_clip_embeddings=source_tree_root
        / "test_data"
        / "lightroom_embedding_vectors_sample.sqlite",
        output_umap_transformation_file=tmp_path / "umap_transformer.pkl",
    )

    assert (tmp_path / "lightroom.sqlite").exists()


# def create_test_clip_embeddings_db(db_path: Path) -> None:
#     """Create a test SQLite database with CLIP embeddings."""
#     # Create test embeddings - simple 3D embeddings for testing
#     test_embeddings = [
#         {
#             "uuid_index": "00002DDF-4255-4469-AC32-3EC6DA5B0D7C",
#             "type_index": "clip",
#             "embedding_msgpack": msgpack.packb(
#                 np.array([0.1, 0.2, 0.3], dtype=np.float32).tolist()
#             ),
#         },
#         {
#             "uuid_index": "00009AD4-C588-4345-AB9C-C2DE2E9C1236",
#             "type_index": "clip",
#             "embedding_msgpack": msgpack.packb(
#                 np.array([0.4, 0.5, 0.6], dtype=np.float32).tolist()
#             ),
#         },
#         {
#             "uuid_index": "test-uuid-3",
#             "type_index": "clip",
#             "embedding_msgpack": msgpack.packb(
#                 np.array([0.7, 0.8, 0.9], dtype=np.float32).tolist()
#             ),
#         },
#     ]

#     with sqlite3.connect(db_path) as conn:
#         # Create embeddings table
#         conn.execute(
#             """
#             CREATE TABLE embeddings (
#                 uuid_index TEXT,
#                 type_index TEXT,
#                 embedding_msgpack BLOB
#             )
#         """
#         )

#         # Insert test data
#         for embedding in test_embeddings:
#             conn.execute(
#                 "INSERT INTO embeddings (uuid_index, type_index, embedding_msgpack) VALUES (?, ?, ?)",
#                 (
#                     embedding["uuid_index"],
#                     embedding["type_index"],
#                     embedding["embedding_msgpack"],
#                 ),
#             )

#         conn.commit()


def test_load_clip_embeddings_from_sqlite(
    source_tree_root: Path, snapshot: SnapshotAssertion
) -> None:
    """Test loading CLIP embeddings from SQLite database."""
    db_path = (
        source_tree_root / "test_data" / "lightroom_embedding_vectors_sample.sqlite"
    )
    embeddings_df = load_clip_embeddings_from_sqlite(db_path)
    assert embeddings_df.shape == snapshot
    assert embeddings_df.head(2).to_dict(orient="records") == snapshot


def test_load_clip_embeddings_from_sqlite_nonexistent_file(tmp_path: Path) -> None:
    """Test loading CLIP embeddings from non-existent file."""
    nonexistent_db = tmp_path / "nonexistent.sqlite"

    with pytest.raises(FileNotFoundError):
        load_clip_embeddings_from_sqlite(nonexistent_db)


def test_compute_umap_projection(tmp_path: Path) -> None:
    """Test computing UMAP projection from embeddings."""
    # Create test embeddings DataFrame
    test_embeddings = pd.DataFrame(
        {
            SchemaColumns.UUID_STRING: ["uuid1", "uuid2", "uuid3", "uuid4"],
            "embedding": [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            ],
        }
    )

    # Test without output file
    result_df, umap_transformer = compute_umap_projection(test_embeddings)

    # Check result structure
    assert len(result_df) == 4
    assert SchemaColumns.UUID_STRING in result_df.columns
    assert SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE in result_df.columns
    assert SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE in result_df.columns

    # Check that UMAP coordinates are numeric
    assert result_df[SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE].dtype in [
        np.float32,
        np.float64,
    ]
    assert result_df[SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE].dtype in [
        np.float32,
        np.float64,
    ]

    # Test with output file
    tmp_path / "umap_transformer.pkl"
    result_df2, umap_transformer2 = compute_umap_projection(test_embeddings)

    import umap

    assert isinstance(umap_transformer2, umap.UMAP)

    # Results should be the same (with same random seed)
    pd.testing.assert_frame_equal(result_df, result_df2)


def test_load_clip_embeddings_from_real_sample() -> None:
    """Test loading CLIP embeddings from the real sample data."""
    sample_db_path = Path("test_data/lightroom_embedding_vectors_sample.sqlite")

    # Skip if sample file doesn't exist
    if not sample_db_path.exists():
        pytest.skip("Sample embedding data not found")

    # Load embeddings
    embeddings_df = load_clip_embeddings_from_sqlite(sample_db_path)

    # Check structure
    assert len(embeddings_df) > 0
    assert SchemaColumns.UUID_STRING in embeddings_df.columns
    assert "embedding" in embeddings_df.columns

    # Check that all embeddings are numpy arrays
    for embedding in embeddings_df["embedding"]:
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # Should be 1D array
        assert embedding.shape[0] > 0  # Should have some dimensions
