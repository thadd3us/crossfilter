"""
Common interface for embedding computation classes.

This module defines the abstract base class that all embedding implementations
should inherit from, ensuring a consistent API across different embedding types.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingInterface(ABC):
    """Abstract base class for embedding computation."""

    @abstractmethod
    def compute_image_embeddings(self, image_paths: list[Path]) -> list[np.ndarray]:
        """
        Compute embeddings for a list of image paths.

        THAD: TODO: No parallel arrays -- take a DataFrame as input, add a column containing the embeddings.  Don't return anything.

        Raises:
            FileNotFoundError: If any image path doesn't exist
            ValueError: If any image cannot be loaded
        """
        pass

    @abstractmethod
    def compute_text_embeddings(self, captions: list[str]) -> list[np.ndarray]:
        """
        Compute embeddings for a list of text captions.

        Args:
            captions: List of text strings to encode

        Returns:
            List of 1D numpy arrays, one embedding vector per input caption
        """
        pass
