"""
Tests for UMAP projection functions.
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
from syrupy import SnapshotAssertion

from crossfilter.inference import umap_haversine
from crossfilter.core.schema import SchemaColumns
from tests.util.syrupy_html_snapshot import HTMLSnapshotExtension

logger = logging.getLogger(__name__)


def _generate_test_embeddings(
    n_samples: int = 100,
    embedding_dim: int = 6,
    missing_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate test data with 3 well-separated clusters."""
    rng = np.random.RandomState(random_state)

    # 3 hardcoded, well-separated centroids
    centers = np.zeros((3, embedding_dim))
    centers[0, 0] = 10.0  # Cluster 0: strong signal in dimension 0
    centers[1, 1] = 10.0  # Cluster 1: strong signal in dimension 1
    centers[2, 2] = 10.0  # Cluster 2: strong signal in dimension 2
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate class assignments
    classes = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]

    # Generate embeddings with small noise around centroids
    noise = rng.normal(
        0, 0.05, (n_samples, embedding_dim)
    )  # Small stddev for tight clusters
    embeddings = centers[classes] + noise
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "UMAP_STRING": [f"sample_{i:03d}" for i in range(n_samples)],
            "FAKE_EMBEDDING_FOR_TESTING_EMBEDDING": list(embeddings),
            "TRUE_CLASS": classes,
        }
    )

    # Set some embeddings to None for missing data
    n_missing = int(n_samples * missing_fraction)
    missing_indices = rng.choice(n_samples, size=n_missing, replace=False)
    df.loc[missing_indices, "FAKE_EMBEDDING_FOR_TESTING_EMBEDDING"] = None

    logger.info(f"Generated {n_samples} samples with {n_missing} missing embeddings")

    return df


def test_run_umap_projection_happy_path(snapshot: SnapshotAssertion) -> None:
    """Test the happy path: UMAP projection with clustering validation."""
    # Generate test data
    df = _generate_test_embeddings(
        n_samples=100,
        embedding_dim=6,
        missing_fraction=0.05,
        random_state=42,
    )

    # Run UMAP projection
    umap_haversine.run_umap_projection(
        df,
        embedding_column="FAKE_EMBEDDING_FOR_TESTING_EMBEDDING",
        output_lat_column=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
        output_lon_column=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
        random_state=42,
    )

    # Plot the results
    fig = px.scatter(
        df,
        y=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
        x=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
        color="TRUE_CLASS",
    )
    html = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html == snapshot(extension_class=HTMLSnapshotExtension)
