"""
SigLIP2 embedding functions for images and text.

References:
- SigLIP2 Blog Post: https://huggingface.co/blog/siglip2
- SigLIP2 Model: https://huggingface.co/google/siglip-so400m-patch14-384
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

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


def compute_image_embeddings(
    image_paths: list[Path],
    batch_size: int = 8,
    model_name: str = "google/siglip-so400m-patch14-384",
) -> list[np.ndarray]:
    """
    Compute SigLIP2 embeddings for a list of image paths.

    The function handles images of any size or aspect ratio by finding the largest
    square region that can be extracted without distortion, then resizing that
    region to the model's expected input size.

    Args:
        image_paths: List of paths to image files
        batch_size: Number of images to process in each batch
        model_name: HuggingFace model identifier for SigLIP2

    Returns:
        List of 1D numpy arrays, one embedding vector per input image

    Raises:
        FileNotFoundError: If any image path doesn't exist
        ValueError: If any image cannot be loaded
    """
    # Handle empty input
    if not image_paths:
        logger.info("No images provided, returning empty list")
        return []

    device = _get_device()
    logger.info(f"Using device: {device}")

    # Load model and processor
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    # TODO: Use torchvision: SiglipImageProcessorFast requires the Torchvision library but it was not found in your environment.
    processor = AutoProcessor.from_pretrained(
        model_name, local_files_only=True, use_fast=False
    )
    model = model.to(device)
    model.eval()

    embeddings = []

    # Process images in batches
    # THAD: TODO: remove the looping logic here -- make this function just for a single batch.
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []

        # Load and preprocess images in the batch
        for image_path in batch_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            try:
                # Load image and convert to RGB if necessary
                image = Image.open(image_path).convert("RGB")

                # Center crop to square to avoid distortion
                image = _center_crop_to_square(image)

                batch_images.append(image)

            except Exception as e:
                raise ValueError(f"Failed to load image {image_path}: {e}")

        # Process batch through model
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt").to(device)

            # Get image embeddings
            outputs = model.get_image_features(**inputs)

            # Normalize embeddings (SigLIP2 typically uses L2 normalization)
            embeddings_normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)

            # Convert to numpy and add to results
            batch_embeddings = embeddings_normalized.cpu().numpy()
            for j in range(len(batch_images)):
                embeddings.append(batch_embeddings[j])

    logger.info(f"Computed embeddings for {len(embeddings)} images")
    return embeddings


def compute_text_embeddings(
    captions: list[str],
    batch_size: int = 32,
    model_name: str = "google/siglip-so400m-patch14-384",
) -> list[np.ndarray]:
    """
    Compute SigLIP2 embeddings for a list of text captions.

    Args:
        captions: List of text strings to encode
        batch_size: Number of captions to process in each batch
        model_name: HuggingFace model identifier for SigLIP2

    Returns:
        List of 1D numpy arrays, one embedding vector per input caption
    """
    # Handle empty input
    if not captions:
        logger.info("No captions provided, returning empty list")
        return []

    device = _get_device()
    logger.info(f"Using device: {device}")

    # Load model and processor
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = model.to(device)
    model.eval()

    embeddings = []

    # Process captions in batches
    # TODO: THAD: Remove the looping logic here -- make this function just be for one batch.
    for i in range(0, len(captions), batch_size):
        batch_captions = captions[i : i + batch_size]

        # Process batch through model
        with torch.no_grad():
            inputs = processor(
                text=batch_captions, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            # Get text embeddings
            outputs = model.get_text_features(**inputs)

            # Normalize embeddings (SigLIP2 typically uses L2 normalization)
            embeddings_normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)

            # Convert to numpy and add to results
            batch_embeddings = embeddings_normalized.cpu().numpy()
            for j in range(len(batch_captions)):
                embeddings.append(batch_embeddings[j])

    logger.info(f"Computed embeddings for {len(embeddings)} captions")
    return embeddings


def generate_captions_from_image_embeddings(
    image_embeddings: list[np.ndarray], batch_size: int = 8, max_length: int = 50
) -> list[str]:
    """
    Generate captions from image embeddings.

    Note: SigLIP2 is primarily designed for contrastive learning rather than
    generative captioning. This function provides a placeholder implementation
    that analyzes embedding characteristics to generate descriptive captions.

    For production use, consider:
    1. Using the original images with a dedicated image-to-text model (e.g., BLIP2)
    2. Training a decoder specifically for SigLIP2 embeddings
    3. Using a multimodal model that supports both embedding and generation

    Args:
        image_embeddings: List of image embedding vectors from SigLIP2
        batch_size: Number of embeddings to process in each batch (for consistency)
        max_length: Maximum length of generated captions (for future use)

    Returns:
        List of generated caption strings
    """
    # Handle empty input
    if not image_embeddings:
        logger.info("No embeddings provided, returning empty list")
        return []

    logger.info(f"Generating captions for {len(image_embeddings)} embeddings")

    captions = []

    for i in range(0, len(image_embeddings), batch_size):
        batch_embeddings = image_embeddings[i : i + batch_size]

        # Analyze embedding characteristics to generate descriptive captions
        for embedding in batch_embeddings:
            # Calculate various embedding statistics
            norm = np.linalg.norm(embedding)
            mean_val = np.mean(embedding)
            std_val = np.std(embedding)
            max_val = np.max(embedding)
            min_val = np.min(embedding)

            # Generate captions based on embedding characteristics
            if norm > 0.95 and std_val > 0.08:
                captions.append(
                    "A complex image with rich visual details and varied features"
                )
            elif norm > 0.9 and mean_val > 0.01:
                captions.append("A detailed image with prominent visual elements")
            elif std_val > 0.1:
                captions.append(
                    "An image with high contrast and diverse visual content"
                )
            elif abs(mean_val) < 0.005 and std_val < 0.05:
                captions.append("A balanced image with subtle visual features")
            elif max_val > 0.2 or min_val < -0.2:
                captions.append("An image with distinctive visual characteristics")
            elif norm > 0.8:
                captions.append("An image with moderate visual complexity")
            else:
                captions.append("A simple image with basic visual elements")

    logger.warning(
        "Caption generation from embeddings uses heuristic analysis. "
        "For production use, consider using original images with a "
        "dedicated image-to-text model like BLIP2 or training a "
        "specific decoder for SigLIP2 embeddings."
    )

    logger.info(f"Generated {len(captions)} captions")
    return captions
