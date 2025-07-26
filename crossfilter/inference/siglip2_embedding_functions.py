"""
SigLIP2 embedding functions for images and text.

References:
- SigLIP2 Blog Post: https://huggingface.co/blog/siglip2
- SigLIP2 Model: https://huggingface.co/google/siglip-so400m-patch14-384

    logger.warning(
        "Caption generation from embeddings uses heuristic analysis. "
        "For production use, consider using original images with a "
        "dedicated image-to-text model like BLIP2 or training a "
        "specific decoder for SigLIP2 embeddings."
    )

"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from crossfilter.inference.embedding_interface import EmbeddingInterface

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Get the appropriate device for computation (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _center_crop_to_square(image: Image.Image) -> Image.Image:
    """
    Center crop the image to the largest square that fits within the original dimensions.
    This ensures no distortion when resizing to the model's expected input size.
    """
    width, height = image.size
    min_dimension = min(width, height)

    # Calculate center crop coordinates
    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = left + min_dimension
    bottom = top + min_dimension

    return image.crop((left, top, right, bottom))


class SigLIP2Embedder(EmbeddingInterface):
    """SigLIP2-based embedding computation class."""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384") -> None:
        """
        Initialize the SigLIP2 embedder with model loading.

        Args:
            model_name: HuggingFace model identifier for SigLIP2
        """
        self.model_name = model_name
        self.device = _get_device()
        logger.info(f"Initializing SigLIP2Embedder with model {model_name} on device: {self.device}")

        # Load model and processor
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        # TODO: Use torchvision: SiglipImageProcessorFast requires the Torchvision library but it was not found in your environment.
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=True, use_fast=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("SigLIP2Embedder initialized successfully")

    def compute_image_embeddings(self, df: pd.DataFrame, image_path_column: str, output_embedding_column: str) -> None:
        """Compute SigLIP2 embeddings for images specified in DataFrame and add them as a new column."""
        # Get image paths, converting to Path objects if needed
        image_paths = df[image_path_column].apply(lambda x: Path(x) if not isinstance(x, Path) else x)
        batch_images = []
        valid_indices = []

        # Load and preprocess all images
        for idx, image_path in image_paths.items():
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            try:
                # Load image and convert to RGB if necessary
                image = Image.open(image_path).convert("RGB")

                # Center crop to square to avoid distortion
                image = _center_crop_to_square(image)

                batch_images.append(image)
                valid_indices.append(idx)

            except Exception as e:
                raise ValueError(f"Failed to load image {image_path}: {e}")

        # Process all images through model
        with torch.no_grad():
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            # Get image embeddings
            outputs = self.model.get_image_features(**inputs)

            # Normalize embeddings (SigLIP2 typically uses L2 normalization)
            embeddings_normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)

            # Convert to numpy
            batch_embeddings = embeddings_normalized.cpu().numpy()

        # Initialize output column with None
        df[output_embedding_column] = None
        
        # Assign embeddings to the correct rows
        for i, idx in enumerate(valid_indices):
            df.at[idx, output_embedding_column] = batch_embeddings[i]

        logger.info(f"Computed embeddings for {len(batch_images)} images")

    def compute_text_embeddings(self, df: pd.DataFrame, text_column: str, output_embedding_column: str) -> None:
        """Compute SigLIP2 embeddings for text specified in DataFrame and add them as a new column."""
        # Get text data, filtering out null values
        text_data = df[text_column].dropna()

        captions = text_data.tolist()
        valid_indices = text_data.index.tolist()

        # Process all captions through model
        with torch.no_grad():
            inputs = self.processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            # Get text embeddings
            outputs = self.model.get_text_features(**inputs)

            # Normalize embeddings (SigLIP2 typically uses L2 normalization)
            embeddings_normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)

            # Convert to numpy
            batch_embeddings = embeddings_normalized.cpu().numpy()

        # Initialize output column with None
        df[output_embedding_column] = None
        
        # Assign embeddings to the correct rows
        for i, idx in enumerate(valid_indices):
            df.at[idx, output_embedding_column] = batch_embeddings[i]

        logger.info(f"Computed embeddings for {len(captions)} captions")


