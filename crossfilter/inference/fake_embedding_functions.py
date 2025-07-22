"""
Fake embedding functions for testing purposes.

This module provides fake embedding functions that compute simple statistics from images
without requiring heavy ML models or network downloads. The fake embeddings are designed
to be fast, deterministic, and suitable for testing.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_image_embeddings(
    image_paths: list[Path],
    batch_size: int = 8,
    model_name: str = "fake-embedding-for-testing",
) -> list[np.ndarray]:
    """
    Compute fake embeddings for a list of image paths by calculating mean and stddev of RGB channels.

    This function provides a fast, deterministic alternative to real embedding models for testing.
    It computes 6-dimensional embeddings: [R_mean, G_mean, B_mean, R_std, G_std, B_std].

    Returns:
        List of 1D numpy arrays, one embedding vector per input image

    Raises:
        FileNotFoundError: If any image path doesn't exist
        ValueError: If any image cannot be loaded
    """
    logger.info(f"Computing fake embeddings for {len(image_paths)} images")

    embeddings = []

    # Process images in batches (for API compatibility)
    # THAD: TODO: Get rid of the baching loop.
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        # Process each image in the batch
        for image_path in batch_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            try:
                # Load image and convert to RGB if necessary
                image = Image.open(image_path).convert("RGB")

                # Convert to numpy array and normalize to [0, 1]
                image_array = np.array(image).astype(np.float32) / 255.0

                # Calculate mean and standard deviation for each RGB channel
                r_channel = image_array[:, :, 0]
                g_channel = image_array[:, :, 1]
                b_channel = image_array[:, :, 2]

                r_mean = np.mean(r_channel)
                g_mean = np.mean(g_channel)
                b_mean = np.mean(b_channel)

                r_std = np.std(r_channel)
                g_std = np.std(g_channel)
                b_std = np.std(b_channel)

                # Create 6-dimensional embedding: [R_mean, G_mean, B_mean, R_std, G_std, B_std]
                embedding = np.array(
                    [r_mean, g_mean, b_mean, r_std, g_std, b_std], dtype=np.float32
                )

                # Normalize the embedding to unit length (to match SIGLIP2 behavior)
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm > 0:
                    embedding = embedding / embedding_norm

                embeddings.append(embedding)

            except Exception as e:
                raise ValueError(f"Failed to load image {image_path}: {e}")

    logger.info(f"Computed fake embeddings for {len(embeddings)} images")
    return embeddings


def compute_text_embeddings(
    captions: list[str],
    batch_size: int = 32,
    model_name: str = "fake-embedding-for-testing",
) -> list[np.ndarray]:
    """
    Compute fake embeddings for a list of text captions.

    This function provides a fast, deterministic alternative to real text embedding models for testing.
    It computes 6-dimensional embeddings based on simple text statistics.

    Returns:
        List of 1D numpy arrays, one embedding vector per input caption
    """
    # Handle empty input
    if not captions:
        logger.info("No captions provided, returning empty list")
        return []

    logger.info(f"Computing fake text embeddings for {len(captions)} captions")

    embeddings = []

    # Process captions in batches (for API compatibility)
    for i in range(0, len(captions), batch_size):
        batch_captions = captions[i : i + batch_size]

        # Process each caption in the batch
        for caption in batch_captions:
            # Calculate simple text statistics
            text_length = len(caption)
            word_count = len(caption.split())
            char_diversity = len(set(caption.lower()))
            avg_word_length = text_length / max(word_count, 1)
            vowel_count = sum(1 for c in caption.lower() if c in "aeiou")
            consonant_count = sum(
                1 for c in caption.lower() if c.isalpha() and c not in "aeiou"
            )

            # Create 6-dimensional embedding from text statistics
            embedding = np.array(
                [
                    text_length / 100.0,  # Normalize text length
                    word_count / 20.0,  # Normalize word count
                    char_diversity / 26.0,  # Normalize character diversity
                    avg_word_length / 10.0,  # Normalize average word length
                    vowel_count / 50.0,  # Normalize vowel count
                    consonant_count / 50.0,  # Normalize consonant count
                ],
                dtype=np.float32,
            )

            # Normalize the embedding to unit length (to match SIGLIP2 behavior)
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm

            embeddings.append(embedding)

    logger.info(f"Computed fake text embeddings for {len(embeddings)} captions")
    return embeddings


def generate_captions_from_image_embeddings(
    image_embeddings: list[np.ndarray], batch_size: int = 8, max_length: int = 50
) -> list[str]:
    """
    Generate fake captions from fake image embeddings.

    Args:
        image_embeddings: List of image embedding vectors
        batch_size: Number of embeddings to process in each batch (for consistency)
        max_length: Maximum length of generated captions (for future use)

    Returns:
        List of generated caption strings
    """
    # Handle empty input
    if not image_embeddings:
        logger.info("No embeddings provided, returning empty list")
        return []

    logger.info(f"Generating fake captions for {len(image_embeddings)} embeddings")

    captions = []

    for i in range(0, len(image_embeddings), batch_size):
        batch_embeddings = image_embeddings[i : i + batch_size]

        # Generate captions based on embedding characteristics
        for embedding in batch_embeddings:
            # For fake embeddings, we know the structure: [R_mean, G_mean, B_mean, R_std, G_std, B_std]
            r_mean, g_mean, b_mean, r_std, g_std, b_std = embedding

            # Generate captions based on color characteristics
            if r_mean > g_mean and r_mean > b_mean:
                if r_std > 0.3:
                    captions.append("An image with varied red tones and high contrast")
                else:
                    captions.append("An image dominated by red colors")
            elif g_mean > r_mean and g_mean > b_mean:
                if g_std > 0.3:
                    captions.append(
                        "An image with varied green tones and natural elements"
                    )
                else:
                    captions.append("An image dominated by green colors")
            elif b_mean > r_mean and b_mean > g_mean:
                if b_std > 0.3:
                    captions.append("An image with varied blue tones like sky or water")
                else:
                    captions.append("An image dominated by blue colors")
            else:
                # Balanced colors
                overall_std = (r_std + g_std + b_std) / 3
                if overall_std > 0.25:
                    captions.append(
                        "A colorful image with balanced tones and high contrast"
                    )
                else:
                    captions.append("A balanced image with uniform lighting")

    logger.info(f"Generated {len(captions)} fake captions")
    return captions
