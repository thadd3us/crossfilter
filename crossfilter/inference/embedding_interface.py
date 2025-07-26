"""
Common interface for embedding computation classes.

This module defines the abstract base class that all embedding implementations
should inherit from, ensuring a consistent API across different embedding types.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class EmbeddingInterface(ABC):
    """Abstract base class for embedding computation."""

    @abstractmethod
    def compute_image_embeddings(self, df: pd.DataFrame, image_path_column: str, output_embedding_column: str) -> None:
        """Compute embeddings for images specified in DataFrame and add them as a new column."""
        pass

    @abstractmethod
    def compute_text_embeddings(self, df: pd.DataFrame, text_column: str, output_embedding_column: str) -> None:
        """Compute embeddings for text specified in DataFrame and add them as a new column."""
        pass
