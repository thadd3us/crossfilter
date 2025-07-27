"""Fake embedding functions for testing purposes.

This module provides fake embedding functions that compute simple statistics from images
without requiring heavy ML models or network downloads. The fake embeddings are designed
to be fast, deterministic, and suitable for testing.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from crossfilter.inference.embedding_interface import EmbeddingInterface

logger = logging.getLogger(__name__)


class FakeEmbedder(EmbeddingInterface):
    """Fake embedding class for testing purposes."""

    def compute_image_embeddings(
        self, df: pd.DataFrame, image_path_column: str, output_embedding_column: str
    ) -> None:
        logger.info(f"Computing fake embeddings for {len(df)=} images")

        df[output_embedding_column] = df[image_path_column].apply(
            _compute_image_embedding
        )
        logger.info(f"Computed fake embeddings for {len(df)} images")

    def compute_text_embeddings(
        self, df: pd.DataFrame, text_column: str, output_embedding_column: str
    ) -> None:
        """Compute fake embeddings for text specified in DataFrame and add them as a new column."""
        # Initialize output column
        df[output_embedding_column] = df[text_column].apply(_compute_text_embedding)
        logger.info(f"Computed fake text embeddings for {len(df)} captions")


def _compute_image_embedding(image_path: Path | None) -> np.ndarray | None:
    """Compute a fake embedding for an image."""
    if not image_path:
        return None

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
        embedding /= embedding_norm

    return embedding


def _compute_text_embedding(text: str | None) -> np.ndarray | None:
    """Compute a fake embedding for text."""
    if not text or len(text) < 5:
        return None

    return np.array([ord(char) for char in text[0:5]], dtype=np.float32)
