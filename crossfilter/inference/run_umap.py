"""
UMAP projection functions for SigLIP2 embeddings with spherical output space.

This module provides functions to run 2D UMAP projections on SigLIP2 embeddings
using spherical embedding space with Haversine distance in the output metric.
This produces coordinates that naturally lie on a sphere, perfect for geographic-style
visualization and analysis.

See: https://umap-learn.readthedocs.io/en/latest/embedding_space.html#spherical-embeddings
"""

import logging

import numpy as np
import pandas as pd

from crossfilter.core.schema import SchemaColumns

logger = logging.getLogger(__name__)


def run_umap_projection(
    df: pd.DataFrame,
    embedding_column: str = "SIGLIP2_EMBEDDING",
    output_lat_column: str = SchemaColumns.SIGLIP2_UMAP2D_HAVERSINE_LATITUDE,
    output_lon_column: str = SchemaColumns.SIGLIP2_UMAP2D_HAVERSINE_LONGITUDE,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> "umap.UMAP":
    """
    Run 2D UMAP projection on SigLIP2 embeddings with spherical embedding space.

    This function takes a DataFrame with embedding vectors and applies UMAP
    dimensionality reduction to project them into 2D coordinates on a sphere
    using Haversine distance in the output space. The resulting coordinates
    are interpreted as latitude/longitude coordinates for geographic-style analysis.

    See: https://umap-learn.readthedocs.io/en/latest/embedding_space.html#spherical-embeddings

    Missing embeddings are handled properly - rows with missing embeddings
    will have NaN values in the output coordinate columns.

    Raises:
        ValueError: If required columns are missing or if no valid embeddings found
        TypeError: If embeddings are not numpy arrays
    """
    import umap


    # Validate required columns exist
    if embedding_column not in df.columns:
        raise ValueError(f"Required column '{embedding_column}' not found in DataFrame")

    # Create result DataFrame with original data
    result_df = df

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
    sphere_mapper = umap.UMAP(
        n_components=2,
        metric="cosine",  # Input space metric (e.g., cosine for normalized embeddings)
        output_metric="haversine",  # Output space metric (haversine for spherical)
        random_state=random_state,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
    ).fit(embedding_matrix)

    # Go through the spherical coordinates and convert to normalized latitude and longitude.
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    latitude = np.degrees(-np.arccos(z)) + 90
    longitude = np.degrees(np.arctan2(x, y))

    result_df.loc[valid_embedding_mask, output_lat_column] = latitude
    result_df.loc[valid_embedding_mask, output_lon_column] = longitude

    result_df[output_lat_column] = result_df[output_lat_column].clip(lower=-90, upper=90)
    result_df[output_lon_column] = result_df[output_lon_column].clip(lower=-180, upper=180)

    logger.info(f"UMAP projection completed. Valid coordinates assigned to {n_valid} rows")

    return sphere_mapper
