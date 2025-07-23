"""
Tests for fake embedding functions.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from syrupy import SnapshotAssertion

from crossfilter.inference.fake_embedding_functions import (
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
    assert len(image_paths) == 10, f"Expected 10 test images, found {len(image_paths)}"

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


def test_compute_image_embeddings(
    test_image_paths: list[Path], snapshot: SnapshotAssertion
) -> None:
    """Test fake image embedding computation."""
    embeddings = compute_image_embeddings(test_image_paths)

    # Verify we got the expected number of embeddings
    assert len(embeddings) == len(test_image_paths)

    # Verify all embeddings are numpy arrays with expected properties
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray), f"Embedding {i} is not a numpy array"
        assert embedding.ndim == 1, f"Embedding {i} is not 1D"
        assert (
            embedding.dtype == np.float32
        ), f"Embedding {i} has wrong dtype: {embedding.dtype}"
        assert (
            len(embedding) == 6
        ), f"Embedding {i} should have 6 dimensions, got {len(embedding)}"

        # Check that embedding is normalized (should have norm close to 1.0)
        norm = np.linalg.norm(embedding)
        assert 0.95 <= norm <= 1.05, f"Embedding {i} norm {norm} is not close to 1.0"

        # Check that values are reasonable (RGB stats should be in [0, 1] range before normalization)
        # After normalization, values can be outside this range, but should be finite
        assert np.all(
            np.isfinite(embedding)
        ), f"Embedding {i} contains non-finite values"

    # Create snapshot data with metadata
    snapshot_data = {
        "num_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "embedding_dtype": str(embeddings[0].dtype),
        "embedding_shapes": [embedding.shape for embedding in embeddings],
        "embedding_norms": [
            round(float(np.linalg.norm(embedding)), 4) for embedding in embeddings
        ],
        # Store all values of each embedding for regression testing (since they're only 6-dimensional)
        "embedding_values": [np.round(embedding, 4).tolist() for embedding in embeddings],
        "image_filenames": [path.name for path in test_image_paths],
    }

    assert snapshot_data == snapshot


def test_compute_text_embeddings(
    test_captions: list[str], snapshot: SnapshotAssertion
) -> None:
    """Test fake text embedding computation for captions."""
    embeddings = compute_text_embeddings(test_captions)

    # Verify we got the expected number of embeddings
    assert len(embeddings) == len(test_captions)

    # Verify all embeddings are numpy arrays with expected properties
    for i, embedding in enumerate(embeddings):
        assert isinstance(
            embedding, np.ndarray
        ), f"Text embedding {i} is not a numpy array"
        assert embedding.ndim == 1, f"Text embedding {i} is not 1D"
        assert embedding.dtype == np.float32, f"Text embedding {i} has wrong dtype"
        assert (
            len(embedding) == 6
        ), f"Text embedding {i} should have 6 dimensions, got {len(embedding)}"

        # Check that embedding is normalized
        norm = np.linalg.norm(embedding)
        assert (
            0.95 <= norm <= 1.05
        ), f"Text embedding {i} norm {norm} is not close to 1.0"

        # Check for finite values
        assert np.all(
            np.isfinite(embedding)
        ), f"Text embedding {i} contains non-finite values"

    # Create snapshot data
    snapshot_data = {
        "num_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "embedding_dtype": str(embeddings[0].dtype),
        "embedding_norms": [
            float(np.linalg.norm(embedding)) for embedding in embeddings
        ],
        # Store all values of each embedding
        "embedding_values": [embedding.tolist() for embedding in embeddings],
        "input_captions": test_captions,
    }

    assert snapshot_data == snapshot


def test_generate_captions_from_image_embeddings(
    test_image_paths: list[Path], snapshot: SnapshotAssertion
) -> None:
    """Test fake caption generation from fake image embeddings."""
    # First compute fake image embeddings
    image_embeddings = compute_image_embeddings(test_image_paths)

    # Generate captions from embeddings
    captions = generate_captions_from_image_embeddings(image_embeddings)

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
    """Test that fake embeddings are consistent across multiple runs."""
    # Use a subset of test images for faster testing
    test_photos_dir = Path(__file__).parent.parent.parent / "test_data" / "test_photos"
    test_paths = sorted(test_photos_dir.glob("*.jpg"))[:3]  # Use first 3 images

    # Compute embeddings twice
    embeddings1 = compute_image_embeddings(test_paths)
    embeddings2 = compute_image_embeddings(test_paths)

    # Verify embeddings are identical (should be deterministic)
    assert len(embeddings1) == len(embeddings2)
    for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
        np.testing.assert_array_almost_equal(
            emb1,
            emb2,
            decimal=5,
            err_msg=f"Fake embeddings not consistent for image {i}",
        )


def test_text_embedding_consistency() -> None:
    """Test that fake text embeddings are consistent across multiple runs."""
    test_texts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "Mountain peaks covered in snow",
    ]

    # Compute embeddings twice
    embeddings1 = compute_text_embeddings(test_texts)
    embeddings2 = compute_text_embeddings(test_texts)

    # Verify embeddings are identical
    assert len(embeddings1) == len(embeddings2)
    for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
        np.testing.assert_array_almost_equal(
            emb1,
            emb2,
            decimal=5,
            err_msg=f"Fake text embeddings not consistent for text {i}",
        )


def test_image_embeddings_deterministic(test_image_paths: list[Path]) -> None:
    """Test that fake embeddings are deterministic across multiple calls."""
    # Take first 6 images for faster testing
    test_paths = test_image_paths[:6]

    # Compute embeddings multiple times
    embeddings1 = compute_image_embeddings(test_paths)
    embeddings2 = compute_image_embeddings(test_paths)
    embeddings3 = compute_image_embeddings(test_paths)

    # All should produce identical results
    assert len(embeddings1) == len(embeddings2) == len(embeddings3)

    for i in range(len(test_paths)):
        np.testing.assert_array_almost_equal(
            embeddings1[i],
            embeddings2[i],
            decimal=5,
            err_msg=f"First vs second run mismatch for image {i}",
        )
        np.testing.assert_array_almost_equal(
            embeddings1[i],
            embeddings3[i],
            decimal=5,
            err_msg=f"First vs third run mismatch for image {i}",
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
