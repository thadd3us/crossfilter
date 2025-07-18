"""
Tests for compute_embeddings_cli.py
"""

import logging
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from syrupy import SnapshotAssertion
from typer.testing import CliRunner

from crossfilter.core.schema import EmbeddingType, SchemaColumns
from crossfilter.inference.compute_embeddings_cli import (
    _compute_embeddings_batch,
    _create_database_schema,
    _get_existing_embeddings,
    _load_embeddings_from_db,
    _scan_image_files,
    _store_umap_model,
    _validate_uuid,
    app,
    main,
)

logger = logging.getLogger(__name__)

# Enable msgpack_numpy for tests
msgpack_numpy.patch()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_embeddings.db"


@pytest.fixture
def temp_input_dir(tmp_path: Path) -> Path:
    """Create a temporary input directory with test images."""
    input_dir = tmp_path / "test_images"
    input_dir.mkdir()
    
    # Create some test UUID.jpg files
    test_uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "6ba7b811-9dad-11d1-80b4-00c04fd430c8"
    ]
    
    nested_uuids = [
        "550e8400-e29b-41d4-a716-446655440001",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c9",
        "6ba7b811-9dad-11d1-80b4-00c04fd430ca"
    ]
    
    for test_uuid in test_uuids:
        # Create fake image files (we'll mock the actual embedding computation)
        image_path = input_dir / f"{test_uuid}.jpg"
        image_path.write_text("fake image content")
    
    # Also create a nested directory structure
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(exist_ok=True)
    for nested_uuid in nested_uuids:
        nested_image_path = nested_dir / f"{nested_uuid}.jpg"
        nested_image_path.write_text("fake nested image content")
    
    # Create some non-UUID files that should be ignored
    (input_dir / "not_a_uuid.jpg").write_text("should be ignored")
    (input_dir / "invalid-uuid-format.jpg").write_text("should be ignored")
    
    return input_dir


@pytest.fixture
def mock_embeddings() -> List[np.ndarray]:
    """Create mock embeddings for testing."""
    return [
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)
    ]


def test_validate_uuid() -> None:
    """Test UUID validation function."""
    # Valid UUIDs
    assert _validate_uuid("550e8400-e29b-41d4-a716-446655440000")
    assert _validate_uuid("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    
    # Invalid UUIDs
    assert not _validate_uuid("not-a-uuid")
    assert not _validate_uuid("550e8400-e29b-41d4-a716")  # Too short
    assert not _validate_uuid("550e8400-e29b-41d4-a716-446655440000-extra")  # Too long
    assert not _validate_uuid("")


def test_scan_image_files(temp_input_dir: Path, snapshot: SnapshotAssertion) -> None:
    """Test scanning for UUID.jpg files."""
    uuid_to_path = _scan_image_files(temp_input_dir)
    
    # Should find 6 valid UUID files (3 in root + 3 in nested)
    assert len(uuid_to_path) == 6
    
    # All keys should be valid UUIDs
    for uuid_str in uuid_to_path.keys():
        assert _validate_uuid(uuid_str)
    
    # All paths should exist
    for path in uuid_to_path.values():
        assert path.exists()
        assert path.suffix == ".jpg"
    
    # Create snapshot data
    snapshot_data = {
        "num_files": len(uuid_to_path),
        "uuids": sorted(uuid_to_path.keys()),
        "relative_paths": [str(path.relative_to(temp_input_dir)) for path in sorted(uuid_to_path.values())]
    }
    
    assert snapshot_data == snapshot


def test_scan_image_files_nonexistent_directory() -> None:
    """Test scanning a non-existent directory."""
    with pytest.raises(FileNotFoundError, match="Input directory not found"):
        _scan_image_files(Path("/nonexistent/directory"))


def test_create_database_schema(temp_db_path: Path) -> None:
    """Test database schema creation."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    
    embeddings_table, umap_model_table = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    # Check that tables were created
    assert embeddings_table.name == "SIGLIP2_EMBEDDINGS"
    assert umap_model_table.name == "SIGLIP2_UMAP_MODEL"
    
    # Check that tables exist in database
    with engine.connect() as conn:
        # Check embeddings table structure
        result = conn.execute(select(embeddings_table.c.UUID).limit(0))
        assert "UUID" in result.keys()
        assert "EMBEDDING" in [col.name for col in embeddings_table.columns]
        
        # Check UMAP model table structure
        result = conn.execute(select(umap_model_table.c.MODEL).limit(0))
        assert "MODEL" in result.keys()


def test_get_existing_embeddings_empty_db(temp_db_path: Path) -> None:
    """Test getting existing embeddings from empty database."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    embeddings_table, _ = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    existing_uuids = _get_existing_embeddings(engine, embeddings_table)
    assert existing_uuids == set()


def test_get_existing_embeddings_with_data(temp_db_path: Path) -> None:
    """Test getting existing embeddings from database with data."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    embeddings_table, _ = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    # Insert some test data
    test_uuids = ["550e8400-e29b-41d4-a716-446655440000", "6ba7b810-9dad-11d1-80b4-00c04fd430c8"]
    test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    with engine.connect() as conn:
        for uuid_str in test_uuids:
            conn.execute(embeddings_table.insert().values(
                UUID=uuid_str,
                EMBEDDING=msgpack.packb(test_embedding)
            ))
        conn.commit()
    
    existing_uuids = _get_existing_embeddings(engine, embeddings_table)
    assert existing_uuids == set(test_uuids)


@patch('crossfilter.inference.compute_embeddings_cli.compute_image_embeddings')
def test_compute_embeddings_batch(mock_compute_embeddings, mock_embeddings: List[np.ndarray]) -> None:
    """Test batch embedding computation."""
    from queue import Queue
    
    # Set up mock
    mock_compute_embeddings.return_value = mock_embeddings
    
    # Create test data
    batch_paths = [Path("/fake/path1.jpg"), Path("/fake/path2.jpg"), Path("/fake/path3.jpg")]
    batch_uuids = ["uuid1", "uuid2", "uuid3"]
    write_queue = Queue()
    batch_size = 3
    
    # Run the function
    _compute_embeddings_batch(batch_paths, batch_uuids, write_queue, batch_size)
    
    # Verify mock was called correctly
    mock_compute_embeddings.assert_called_once_with(batch_paths, batch_size=batch_size)
    
    # Verify items were added to queue
    assert write_queue.qsize() == 3
    
    # Check queue contents
    for i in range(3):
        uuid_str, embedding = write_queue.get()
        assert uuid_str == batch_uuids[i]
        np.testing.assert_array_equal(embedding, mock_embeddings[i])


def test_load_embeddings_from_db_empty(temp_db_path: Path) -> None:
    """Test loading embeddings from empty database."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    embeddings_table, _ = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    df = _load_embeddings_from_db(engine, embeddings_table, EmbeddingType.SIGLIP2)
    
    assert len(df) == 0
    assert list(df.columns) == [SchemaColumns.UUID_STRING, "SIGLIP2_EMBEDDING"]


def test_load_embeddings_from_db_with_data(temp_db_path: Path, snapshot: SnapshotAssertion) -> None:
    """Test loading embeddings from database with data."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    embeddings_table, _ = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    # Insert test data
    test_data = [
        ("550e8400-e29b-41d4-a716-446655440000", np.array([0.1, 0.2, 0.3], dtype=np.float32)),
        ("6ba7b810-9dad-11d1-80b4-00c04fd430c8", np.array([0.4, 0.5, 0.6], dtype=np.float32))
    ]
    
    with engine.connect() as conn:
        for uuid_str, embedding in test_data:
            conn.execute(embeddings_table.insert().values(
                UUID=uuid_str,
                EMBEDDING=msgpack.packb(embedding)
            ))
        conn.commit()
    
    df = _load_embeddings_from_db(engine, embeddings_table, EmbeddingType.SIGLIP2)
    
    assert len(df) == 2
    assert list(df.columns) == [SchemaColumns.UUID_STRING, "SIGLIP2_EMBEDDING"]
    
    # Create snapshot data
    snapshot_data = {
        "num_rows": len(df),
        "uuids": sorted(df[SchemaColumns.UUID_STRING].tolist()),
        "embedding_shapes": [emb.shape for emb in df["SIGLIP2_EMBEDDING"]],
        "embedding_dtypes": [str(emb.dtype) for emb in df["SIGLIP2_EMBEDDING"]],
        "embedding_samples": [emb[:3].tolist() for emb in df["SIGLIP2_EMBEDDING"]]
    }
    
    assert snapshot_data == snapshot


def test_store_umap_model(temp_db_path: Path) -> None:
    """Test storing UMAP model in database."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    _, umap_model_table = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    # Create a mock UMAP model
    mock_model = {"n_components": 2, "n_neighbors": 15, "embedding_": np.array([[0.1, 0.2], [0.3, 0.4]])}
    
    # Store the model
    _store_umap_model(engine, umap_model_table, mock_model)
    
    # Verify it was stored
    with engine.connect() as conn:
        result = conn.execute(select(umap_model_table.c.MODEL))
        rows = result.fetchall()
        
        assert len(rows) == 1
        stored_model = pickle.loads(rows[0][0])
        assert stored_model["n_components"] == 2
        assert stored_model["n_neighbors"] == 15
        np.testing.assert_array_equal(stored_model["embedding_"], mock_model["embedding_"])


def test_store_umap_model_overwrites_existing(temp_db_path: Path) -> None:
    """Test that storing UMAP model overwrites existing model."""
    engine = create_engine(f"sqlite:///{temp_db_path}")
    _, umap_model_table = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    # Store first model
    model1 = {"version": 1}
    _store_umap_model(engine, umap_model_table, model1)
    
    # Store second model
    model2 = {"version": 2}
    _store_umap_model(engine, umap_model_table, model2)
    
    # Verify only the second model exists
    with engine.connect() as conn:
        result = conn.execute(select(umap_model_table.c.MODEL))
        rows = result.fetchall()
        
        assert len(rows) == 1
        stored_model = pickle.loads(rows[0][0])
        assert stored_model["version"] == 2


@patch('crossfilter.inference.compute_embeddings_cli.compute_image_embeddings')
def test_cli_basic_functionality(mock_compute_embeddings, temp_input_dir: Path, temp_db_path: Path) -> None:
    """Test basic CLI functionality."""
    # Set up mock
    mock_embeddings = [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in range(6)]
    mock_compute_embeddings.return_value = mock_embeddings
    
    runner = CliRunner()
    
    # Run the CLI
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(temp_input_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--no-reproject_umap_embeddings"
    ])
    
    # Check that command succeeded
    assert result.exit_code == 0
    
    # Verify database was created and has expected data
    engine = create_engine(f"sqlite:///{temp_db_path}")
    embeddings_table, _ = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    existing_uuids = _get_existing_embeddings(engine, embeddings_table)
    assert len(existing_uuids) == 6  # Should have processed 6 UUID files


@patch('crossfilter.inference.compute_embeddings_cli.compute_image_embeddings')
@patch('crossfilter.inference.compute_embeddings_cli.run_umap_projection')
def test_cli_with_umap_projection(mock_run_umap, mock_compute_embeddings, temp_input_dir: Path, temp_db_path: Path) -> None:
    """Test CLI with UMAP projection enabled."""
    # Set up mocks
    mock_embeddings = [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in range(6)]
    mock_compute_embeddings.return_value = mock_embeddings
    
    # Create a simple picklable mock UMAP model using a dictionary
    mock_umap_model = {
        'n_components': 2,
        'embedding_': np.array([[0.1, 0.2], [0.3, 0.4]])
    }
    mock_run_umap.return_value = mock_umap_model
    
    runner = CliRunner()
    
    # Run the CLI with UMAP projection
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(temp_input_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--no-recompute_existing_embeddings",
        "--reproject_umap_embeddings"
    ])
    
    # Check that command succeeded
    assert result.exit_code == 0
    
    # Verify UMAP projection was called
    mock_run_umap.assert_called_once()
    
    # Verify UMAP model was stored
    engine = create_engine(f"sqlite:///{temp_db_path}")
    _, umap_model_table = _create_database_schema(engine, EmbeddingType.SIGLIP2)
    
    with engine.connect() as conn:
        result = conn.execute(select(umap_model_table.c.MODEL))
        rows = result.fetchall()
        assert len(rows) == 1


def test_cli_invalid_input_directory() -> None:
    """Test CLI with invalid input directory."""
    runner = CliRunner()
    
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", "/nonexistent/directory",
        "--output_embeddings_db", "/tmp/test.db",
        "--batch_size", "2"
    ])
    
    # Should fail with BadParameter
    assert result.exit_code != 0


@patch('crossfilter.inference.compute_embeddings_cli.compute_image_embeddings')
def test_cli_recompute_existing_embeddings(mock_compute_embeddings, temp_input_dir: Path, temp_db_path: Path) -> None:
    """Test CLI with recompute_existing_embeddings=true."""
    # Set up mock
    mock_embeddings = [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in range(6)]
    mock_compute_embeddings.return_value = mock_embeddings
    
    runner = CliRunner()
    
    # Run CLI first time
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(temp_input_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--no-recompute_existing_embeddings",
        "--no-reproject_umap_embeddings"
    ])
    assert result.exit_code == 0
    
    # Reset mock call count
    mock_compute_embeddings.reset_mock()
    
    # Run CLI second time with recompute=false (should not recompute)
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(temp_input_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--no-recompute_existing_embeddings",
        "--no-reproject_umap_embeddings"
    ])
    assert result.exit_code == 0
    assert mock_compute_embeddings.call_count == 0  # Should not be called
    
    # Run CLI third time with recompute=true (should recompute)
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(temp_input_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--recompute_existing_embeddings",
        "--no-reproject_umap_embeddings"
    ])
    assert result.exit_code == 0
    assert mock_compute_embeddings.call_count > 0  # Should be called again


def test_cli_no_images_found(tmp_path: Path, temp_db_path: Path) -> None:
    """Test CLI when no valid UUID images are found."""
    # Create empty input directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    runner = CliRunner()
    
    result = runner.invoke(app, [
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(empty_dir),
        "--output_embeddings_db", str(temp_db_path),
        "--batch_size", "2",
        "--no-reproject_umap_embeddings"
    ])
    
    # Should succeed but warn about no images
    assert result.exit_code == 0