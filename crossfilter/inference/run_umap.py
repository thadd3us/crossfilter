"""
UMAP projection functions for SigLIP2 embeddings with spherical output space.

This module provides functions to run 2D UMAP projections on SigLIP2 embeddings
using spherical embedding space with Haversine distance in the output metric.
This produces coordinates that naturally lie on a sphere, perfect for geographic-style
visualization and analysis.

See: https://umap-learn.readthedocs.io/en/latest/embedding_space.html#spherical-embeddings
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import umap

from crossfilter.core.schema import SchemaColumns

logger = logging.getLogger(__name__)


def run_umap_projection(
    df: pd.DataFrame,
    embedding_column: str = "SIGLIP2_EMBEDDING",
    umap_string_column: str = "UMAP_STRING",
    output_lat_column: str = SchemaColumns.SIGLIP2_UMAP2D_HAVERSINE_LATITUDE,
    output_lon_column: str = SchemaColumns.SIGLIP2_UMAP2D_HAVERSINE_LONGITUDE,
    n_components: int = 2,
    metric: str = "cosine",
    output_metric: str = "haversine",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame:
    """
    Run 2D UMAP projection on SigLIP2 embeddings with spherical embedding space.
    
    This function takes a DataFrame with embedding vectors and applies UMAP
    dimensionality reduction to project them into 2D coordinates on a sphere
    using Haversine distance in the output space. The resulting coordinates
    are interpreted as latitude/longitude coordinates for geographic-style analysis.
    
    See: https://umap-learn.readthedocs.io/en/latest/embedding_space.html#spherical-embeddings
    
    Missing embeddings are handled properly - rows with missing embeddings
    will have NaN values in the output coordinate columns.
    
    Args:
        df: Input DataFrame containing embedding vectors
        embedding_column: Name of column containing 1D numpy arrays of embeddings
        umap_string_column: Name of column containing UMAP string identifiers
        output_lat_column: Name of output column for latitude coordinates
        output_lon_column: Name of output column for longitude coordinates
        n_components: Number of dimensions for UMAP output (should be 2)
        metric: Distance metric for input space (e.g., "cosine", "euclidean", "manhattan")
        output_metric: Distance metric for output space ("haversine" for spherical embeddings)
        random_state: Random seed for reproducibility
        n_neighbors: Number of neighbors parameter for UMAP
        min_dist: Minimum distance parameter for UMAP
        
    Returns:
        DataFrame with original columns plus new latitude/longitude columns
        
    Raises:
        ValueError: If required columns are missing or if no valid embeddings found
        TypeError: If embeddings are not numpy arrays
    """
    if df.empty:
        logger.info("Empty DataFrame provided, returning with NaN coordinates")
        result_df = df.copy()
        result_df[output_lat_column] = np.nan
        result_df[output_lon_column] = np.nan
        return result_df
    
    # Validate required columns exist
    if embedding_column not in df.columns:
        raise ValueError(f"Required column '{embedding_column}' not found in DataFrame")
    if umap_string_column not in df.columns:
        raise ValueError(f"Required column '{umap_string_column}' not found in DataFrame")
    
    # Create result DataFrame with original data
    result_df = df.copy()
    
    # Initialize output columns with NaN
    result_df[output_lat_column] = np.nan
    result_df[output_lon_column] = np.nan
    
    # Find rows with valid (non-null) embeddings
    valid_embedding_mask = df[embedding_column].notna()
    
    if not valid_embedding_mask.any():
        logger.warning("No valid embeddings found, all coordinates will be NaN")
        return result_df
    
    # Extract valid embeddings using pandas operations
    valid_embeddings_series = df.loc[valid_embedding_mask, embedding_column]
    
    # Validate first embedding to check type and shape
    first_embedding = valid_embeddings_series.iloc[0]
    if not isinstance(first_embedding, np.ndarray):
        raise TypeError(f"Embeddings must be numpy arrays, got {type(first_embedding)}")
    if first_embedding.ndim != 1:
        raise ValueError(f"Embeddings must be 1D, got shape {first_embedding.shape}")
    
    # Stack all valid embeddings into a matrix using pandas operations
    embedding_matrix = np.stack(valid_embeddings_series.values)
    
    logger.info(f"Running UMAP on {len(valid_embeddings_series)} valid embeddings "
                f"with shape {embedding_matrix.shape}")
    
    # Configure and run UMAP
    # Ensure n_neighbors is valid (must be >= 2 and < n_samples)
    n_valid = len(valid_embeddings_series)
    effective_n_neighbors = min(n_neighbors, n_valid - 1)
    effective_n_neighbors = max(2, effective_n_neighbors)  # UMAP requires n_neighbors >= 2
    
    # Handle case where we have only 1 embedding (UMAP can't run)
    if n_valid == 1:
        logger.warning("Only 1 valid embedding found, assigning center coordinates")
        # Use pandas operations to assign center coordinates
        result_df.loc[valid_embedding_mask, output_lat_column] = 0.0
        result_df.loc[valid_embedding_mask, output_lon_column] = 0.0
        return result_df
    
    # Configure UMAP with spherical embedding space using Haversine distance
    # See: https://umap-learn.readthedocs.io/en/latest/embedding_space.html#spherical-embeddings
    umap_reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,  # Input space metric (e.g., cosine for normalized embeddings)
        output_metric=output_metric,  # Output space metric (haversine for spherical)
        random_state=random_state,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
    )
    
    # Fit and transform embeddings
    umap_coords = umap_reducer.fit_transform(embedding_matrix)
    
    # Normalize UMAP coordinates to valid lat/lon ranges
    # IMPORTANT: Use normalization instead of clipping to preserve clustering structure.
    # Clipping would force many points to the same extreme coordinates, destroying clusters.
    lat_coords, lon_coords = _convert_umap_to_latlon(umap_coords)
    
    # Assign coordinates back using pandas operations (no for loop)
    result_df.loc[valid_embedding_mask, output_lat_column] = lat_coords
    result_df.loc[valid_embedding_mask, output_lon_column] = lon_coords
    
    logger.info(f"UMAP projection completed. Valid coordinates assigned to {n_valid} rows")
    
    return result_df


def _convert_umap_to_latlon(umap_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert UMAP 2D coordinates to latitude/longitude ranges.
    
    UMAP coordinates are typically in an arbitrary range. We normalize them
    to valid latitude [-90, 90] and longitude [-180, 180] ranges.
    
    Args:
        umap_coords: UMAP output coordinates with shape (n_samples, 2)
        
    Returns:
        Tuple of (latitude_array, longitude_array)
    """
    if umap_coords.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {umap_coords.shape}")
    
    # Normalize coordinates to [0, 1] range
    min_vals = umap_coords.min(axis=0)
    max_vals = umap_coords.max(axis=0)
    
    # Handle case where all coordinates are the same (no variation)
    coord_ranges = max_vals - min_vals
    coord_ranges = np.where(coord_ranges == 0, 1, coord_ranges)  # Avoid division by zero
    
    normalized_coords = (umap_coords - min_vals) / coord_ranges
    
    # Map to latitude [-90, 90] and longitude [-180, 180]
    latitudes = normalized_coords[:, 0] * 180 - 90  # Map [0,1] to [-90, 90]
    longitudes = normalized_coords[:, 1] * 360 - 180  # Map [0,1] to [-180, 180]
    
    return latitudes, longitudes