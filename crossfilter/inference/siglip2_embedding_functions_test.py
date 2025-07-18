"""
Tests for SigLIP2 embedding functions.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from syrupy import SnapshotAssertion

from crossfilter.inference.siglip2_embedding_functions import (
    compute_image_embeddings,
    compute_text_embeddings,
    generate_captions_from_image_embeddings,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def test_image_paths() -> list[Path]:
    """Get paths to all test photos."""
    test_photos_dir = Path(__file__).parent.parent.parent / "test_data" / "test_photos"

    # Get all jpg files in the test_photos directory
    image_paths = sorted(test_photos_dir.glob("*.jpg"))

    # Verify we have the expected test images
    assert len(image_paths) == 9, f"Expected 9 test images, found {len(image_paths)}"

    # Verify all files exist
    for path in image_paths:
        assert path.exists(), f"Test image not found: {path}"

    return image_paths


@pytest.fixture
def test_captions() -> list[str]:
    """Sample captions for testing text embeddings."""
    return [
        "A beautiful golden gate bridge in San Francisco",
        "Planet Earth as seen from space showing blue oceans and white clouds",
        "A peaceful forest path with tall trees on both sides",
        "Stunning mountain landscape with snow-capped peaks",
        "Modern city skyline with tall skyscrapers at sunset",
        "Close-up view of colorful flowers in nature",
        "Busy street scene with people and vehicles",
        "Abstract geometric pattern with vibrant colors",
        "Urban architectural details and building textures",
    ]


def test_compute_image_embeddings_single_batch(test_image_paths: list[Path], snapshot: SnapshotAssertion) -> None:
    """Test image embedding computation with all images in a single batch."""
    # Use a large batch size to process all images at once
    embeddings = compute_image_embeddings(test_image_paths, batch_size=16)

    # Verify we got the expected number of embeddings
    assert len(embeddings) == len(test_image_paths)

    # Verify all embeddings are numpy arrays with expected properties
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray), f"Embedding {i} is not a numpy array"
        assert embedding.ndim == 1, f"Embedding {i} is not 1D"
        assert embedding.dtype == np.float32, f"Embedding {i} has wrong dtype: {embedding.dtype}"
        assert len(embedding) > 0, f"Embedding {i} is empty"

        # Check that embedding is normalized (should have norm close to 1.0)
        norm = np.linalg.norm(embedding)
        assert 0.95 <= norm <= 1.05, f"Embedding {i} norm {norm} is not close to 1.0"

    # Create snapshot data with metadata
    snapshot_data = {
        "num_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "embedding_dtype": str(embeddings[0].dtype),
        "embedding_shapes": [embedding.shape for embedding in embeddings],
        "embedding_norms": [float(np.linalg.norm(embedding)) for embedding in embeddings],
        # Store first few values of each embedding for regression testing
        "embedding_samples": [embedding[:5].tolist() for embedding in embeddings],
        "image_filenames": [path.name for path in test_image_paths],
    }

    assert snapshot_data == snapshot


def test_compute_image_embeddings_small_batches(test_image_paths: list[Path], snapshot: SnapshotAssertion) -> None:
    """Test image embedding computation with small batch sizes."""
    # Use small batch size to test batching logic
    embeddings = compute_image_embeddings(test_image_paths, batch_size=3)

    # Verify we got the expected number of embeddings
    assert len(embeddings) == len(test_image_paths)

    # Verify all embeddings are properly formed
    for embedding in embeddings:
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) > 0

    # Create snapshot data
    snapshot_data = {
        "num_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "batch_size_used": 3,
        # Store first few values for consistency check
        "embedding_samples": [embedding[:3].tolist() for embedding in embeddings],
    }

    assert snapshot_data == snapshot


def test_compute_text_embeddings(test_captions: list[str], snapshot: SnapshotAssertion) -> None:
    """Test text embedding computation for captions."""
    embeddings = compute_text_embeddings(test_captions, batch_size=4)

    # Verify we got the expected number of embeddings
    assert len(embeddings) == len(test_captions)

    # Verify all embeddings are numpy arrays with expected properties
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray), f"Text embedding {i} is not a numpy array"
        assert embedding.ndim == 1, f"Text embedding {i} is not 1D"
        assert embedding.dtype == np.float32, f"Text embedding {i} has wrong dtype"
        assert len(embedding) > 0, f"Text embedding {i} is empty"

        # Check that embedding is normalized
        norm = np.linalg.norm(embedding)
        assert 0.95 <= norm <= 1.05, f"Text embedding {i} norm {norm} is not close to 1.0"

    # Create snapshot data
    snapshot_data = {
        "num_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "embedding_dtype": str(embeddings[0].dtype),
        "embedding_norms": [float(np.linalg.norm(embedding)) for embedding in embeddings],
        # Store first few values of each embedding
        "embedding_samples": [embedding[:5].tolist() for embedding in embeddings],
        "input_captions": test_captions,
    }

    assert snapshot_data == snapshot


def test_generate_captions_from_image_embeddings(test_image_paths: list[Path], snapshot: SnapshotAssertion) -> None:
    """Test caption generation from image embeddings."""
    # First compute image embeddings
    image_embeddings = compute_image_embeddings(test_image_paths, batch_size=8)

    # Generate captions from embeddings
    captions = generate_captions_from_image_embeddings(image_embeddings, batch_size=4)

    # Verify we got the expected number of captions
    assert len(captions) == len(image_embeddings)

    # Verify all captions are strings
    for i, caption in enumerate(captions):
        assert isinstance(caption, str), f"Caption {i} is not a string"
        assert len(caption) > 0, f"Caption {i} is empty"

    # Create snapshot data
    snapshot_data = {
        "num_captions": len(captions),
        "captions": captions,
        "image_filenames": [path.name for path in test_image_paths],
        "caption_lengths": [len(caption) for caption in captions],
    }

    assert snapshot_data == snapshot


def test_embedding_consistency() -> None:
    """Test that embeddings are consistent across multiple runs."""
    # Use a subset of test images for faster testing
    test_photos_dir = Path(__file__).parent.parent.parent / "test_data" / "test_photos"
    test_paths = sorted(test_photos_dir.glob("*.jpg"))[:3]  # Use first 3 images

    # Compute embeddings twice
    embeddings1 = compute_image_embeddings(test_paths, batch_size=2)
    embeddings2 = compute_image_embeddings(test_paths, batch_size=2)

    # Verify embeddings are identical (should be deterministic)
    assert len(embeddings1) == len(embeddings2)
    for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
        np.testing.assert_array_almost_equal(
            emb1, emb2, decimal=5,
            err_msg=f"Embeddings not consistent for image {i}"
        )


def test_text_embedding_consistency() -> None:
    """Test that text embeddings are consistent across multiple runs."""
    test_texts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "Mountain peaks covered in snow",
    ]

    # Compute embeddings twice
    embeddings1 = compute_text_embeddings(test_texts, batch_size=2)
    embeddings2 = compute_text_embeddings(test_texts, batch_size=2)

    # Verify embeddings are identical
    assert len(embeddings1) == len(embeddings2)
    for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
        np.testing.assert_array_almost_equal(
            emb1, emb2, decimal=5,
            err_msg=f"Text embeddings not consistent for text {i}"
        )


def test_image_embeddings_different_batch_sizes(test_image_paths: list[Path]) -> None:
    """Test that different batch sizes produce identical embeddings."""
    # Take first 6 images for faster testing
    test_paths = test_image_paths[:6]

    # Compute with different batch sizes
    embeddings_batch2 = compute_image_embeddings(test_paths, batch_size=2)
    embeddings_batch3 = compute_image_embeddings(test_paths, batch_size=3)
    embeddings_batch6 = compute_image_embeddings(test_paths, batch_size=6)

    # All should produce identical results
    assert len(embeddings_batch2) == len(embeddings_batch3) == len(embeddings_batch6)

    for i in range(len(test_paths)):
        np.testing.assert_array_almost_equal(
            embeddings_batch2[i], embeddings_batch3[i], decimal=5,
            err_msg=f"Batch size 2 vs 3 mismatch for image {i}"
        )
        np.testing.assert_array_almost_equal(
            embeddings_batch2[i], embeddings_batch6[i], decimal=5,
            err_msg=f"Batch size 2 vs 6 mismatch for image {i}"
        )


def test_error_handling_missing_image() -> None:
    """Test error handling for missing image files."""
    missing_path = Path("/nonexistent/image.jpg")

    with pytest.raises(FileNotFoundError, match="Image not found"):
        compute_image_embeddings([missing_path])


def test_error_handling_invalid_image(tmp_path: Path) -> None:
    """Test error handling for invalid image files."""
    # Create a non-image file
    invalid_file = tmp_path / "not_an_image.jpg"
    invalid_file.write_text("This is not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        compute_image_embeddings([invalid_file])


def test_empty_inputs() -> None:
    """Test handling of empty input lists."""
    # Empty image list
    image_embeddings = compute_image_embeddings([])
    assert image_embeddings == []

    # Empty text list
    text_embeddings = compute_text_embeddings([])
    assert text_embeddings == []

    # Empty embeddings for caption generation
    captions = generate_captions_from_image_embeddings([])
    assert captions == []
