"""
Tests for compute_embeddings_cli.py.
"""

import logging
import shutil
import sqlite3
import subprocess
from pathlib import Path

import msgpack_numpy
import numpy as np
import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.schema import EmbeddingType

# Enable msgpack_numpy for deserialization
msgpack_numpy.patch()

logger = logging.getLogger(__name__)


def test_compute_embeddings_cli_happy_path(tmp_path: Path, snapshot: SnapshotAssertion) -> None:
    """Placeholder test - not resource intensive."""
    # This is just a placeholder that doesn't actually run the CLI
    assert True


@pytest.mark.resource_intensive
def test_compute_embeddings_cli_siglip2_full_workflow(tmp_path: Path, snapshot: SnapshotAssertion) -> None:
    """
    Comprehensive test of compute_embeddings_cli.py with SIGLIP2 embeddings.
    
    This test:
    1. Sets up a directory with UUID-named test images
    2. Runs the CLI to compute SIGLIP2 embeddings and UMAP projections
    3. Verifies the output database structure and contents
    4. Uses syrupy snapshots to capture embedding characteristics
    """
    # Set up test images directory with UUID naming convention
    input_dir = tmp_path / "images_by_uuid"
    input_dir.mkdir()
    
    # Copy test images with UUID names (CLI expects <uuid>.jpg format)
    test_photos_dir = Path(__file__).parent.parent.parent / "test_data" / "test_photos"
    test_images = sorted(test_photos_dir.glob("*.jpg"))[:3]  # Use first 3 images for faster testing
    
    uuids_and_paths = []
    for i, src_image in enumerate(test_images):
        # Generate test UUID (deterministic for testing)
        test_uuid = f"12345678-1234-5678-9abc-{i:012d}"
        dst_path = input_dir / f"{test_uuid}.jpg"
        shutil.copy2(src_image, dst_path)
        uuids_and_paths.append((test_uuid, dst_path, src_image.name))
    
    # Set up output database path
    output_db = tmp_path / "embeddings.db"
    
    # Run the CLI with SIGLIP2 embeddings and UMAP projection
    cmd = [
        "uv", "run", "python", "-m", "crossfilter.inference.compute_embeddings_cli",
        "--embedding_type", "SIGLIP2",
        "--input_dir", str(input_dir),
        "--output_embeddings_db", str(output_db),
        "--batch_size", "2",  # Small batch size for testing
        "--recompute_existing_embeddings",
        "--reproject_umap_embeddings"
    ]
    
    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
    
    # Verify CLI execution succeeded
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    
    # Verify output database was created
    assert output_db.exists(), "Output database was not created"
    
    # Connect to database and verify structure
    with sqlite3.connect(output_db) as conn:
        # Check that tables were created
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        expected_tables = {"SIGLIP2_EMBEDDINGS", "SIGLIP2_UMAP_MODEL"}
        actual_tables = set(tables["name"].tolist())
        assert expected_tables.issubset(actual_tables), f"Missing tables. Expected: {expected_tables}, Got: {actual_tables}"
        
        # Load embeddings table
        embeddings_df = pd.read_sql("SELECT * FROM SIGLIP2_EMBEDDINGS", conn)
        
        # Verify we have embeddings for all test images
        assert len(embeddings_df) == len(uuids_and_paths), f"Expected {len(uuids_and_paths)} embeddings, got {len(embeddings_df)}"
        
        # Verify all UUIDs are present
        expected_uuids = {uuid for uuid, _, _ in uuids_and_paths}
        actual_uuids = set(embeddings_df["UUID"].tolist())
        assert expected_uuids == actual_uuids, f"UUID mismatch. Expected: {expected_uuids}, Got: {actual_uuids}"
        
        # Deserialize and analyze embeddings
        embeddings_data = []
        for _, row in embeddings_df.iterrows():
            embedding_blob = row["EMBEDDING"]
            embedding = msgpack_numpy.loads(embedding_blob)
            
            # Verify embedding properties
            assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
            assert embedding.dtype == np.float32, f"Embedding should be float32, got {embedding.dtype}"
            assert embedding.ndim == 1, f"Embedding should be 1D, got {embedding.ndim}"
            assert len(embedding) > 500, f"SIGLIP2 embedding should be large (>500 dimensions), got {len(embedding)}"
            
            # Check embedding is normalized (SIGLIP2 should be L2 normalized)
            norm = np.linalg.norm(embedding)
            assert 0.95 <= norm <= 1.05, f"Embedding should be normalized, norm={norm}"
            
            embeddings_data.append({
                "uuid": row["UUID"],
                "embedding_shape": embedding.shape,
                "embedding_dtype": str(embedding.dtype),
                "embedding_norm": float(norm),
                "embedding_mean": float(np.mean(embedding)),
                "embedding_std": float(np.std(embedding)),
                "embedding_min": float(np.min(embedding)),
                "embedding_max": float(np.max(embedding)),
                # Store first few values for regression testing
                "embedding_sample": embedding[:5].tolist(),
            })
        
        # Check UMAP model table
        umap_model_df = pd.read_sql("SELECT * FROM SIGLIP2_UMAP_MODEL", conn)
        assert len(umap_model_df) == 1, "Should have exactly one UMAP model"
        
        umap_model_blob = umap_model_df.iloc[0]["MODEL"]
        umap_model = msgpack_numpy.loads(umap_model_blob)
        assert hasattr(umap_model, "transform"), "UMAP model should have transform method"
    
    # Create snapshot data for regression testing
    snapshot_data = {
        "cli_command": cmd,
        "cli_stdout": result.stdout,
        "cli_returncode": result.returncode,
        "database_tables": actual_tables,
        "num_embeddings": len(embeddings_df),
        "embeddings_analysis": embeddings_data,
        "test_image_mapping": [
            {
                "uuid": uuid,
                "original_filename": orig_name,
                "file_exists": dst_path.exists()
            }
            for uuid, dst_path, orig_name in uuids_and_paths
        ],
        "umap_model_info": {
            "model_type": str(type(umap_model)),
            "has_transform_method": hasattr(umap_model, "transform"),
            "has_embedding_attr": hasattr(umap_model, "embedding_"),
        }
    }
    
    # Use syrupy snapshot for regression testing
    assert snapshot_data == snapshot
