"""Tests for CLIP embedding plot module."""

from pathlib import Path

import pandas as pd
import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension

from crossfilter.core.clip_embedding_projection_state import (
    ClipEmbeddingProjectionState,
)
from crossfilter.core.schema import SchemaColumns, load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.clip_embedding_plot import create_clip_embedding_plot


class HTMLSnapshotExtension(SingleFileSnapshotExtension):
    """Custom syrupy extension to save HTML files with .html extension."""

    _file_extension = "html"

    def serialize(self, data, **kwargs) -> bytes:
        """Serialize string data to bytes for file storage."""
        if isinstance(data, str):
            return data.encode("utf-8")
        return super().serialize(data, **kwargs)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load sample data for testing."""
    sample_path = Path(__file__).parent.parent.parent / "test_data" / "sample_100.jsonl"
    return load_jsonl_to_dataframe(sample_path)


@pytest.fixture
def session_state_with_data(sample_data: pd.DataFrame) -> SessionState:
    """Create session state with sample data loaded."""
    session_state = SessionState()
    session_state.load_dataframe(sample_data)
    return session_state


@pytest.fixture
def individual_clip_data(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Create individual CLIP embedding data for testing."""
    individual_df = sample_data.copy()
    # Add CLIP UMAP coordinates for testing
    individual_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE] = individual_df[
        SchemaColumns.GPS_LATITUDE
    ]
    individual_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE] = individual_df[
        SchemaColumns.GPS_LONGITUDE
    ]
    return individual_df


@pytest.fixture
def grouped_clip_data(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Create grouped CLIP embedding data with COUNT column."""
    grouped_df = sample_data.copy()
    # Add CLIP UMAP coordinates for testing
    grouped_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE] = grouped_df[
        SchemaColumns.GPS_LATITUDE
    ]
    grouped_df[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE] = grouped_df[
        SchemaColumns.GPS_LONGITUDE
    ]
    # Add some COUNT values for testing marker sizing
    count_values = [1, 5, 10, 20, 50] * (len(grouped_df) // 5 + 1)
    grouped_df[SchemaColumns.COUNT] = count_values[: len(grouped_df)]
    return grouped_df


@pytest.fixture
def clip_embedding_space_data() -> pd.DataFrame:
    """Create test data with CLIP embeddings projected into semantic space."""
    import datetime

    # Create points in CLIP embedding space (different from GPS coordinates)
    # These represent semantic similarity rather than geographic proximity
    data = []
    locations = [
        ("Nature Photo", 45.0, -120.0),  # Clustered in one area of semantic space
        ("Landscape Photo", 45.1, -120.1),
        ("Mountain Photo", 45.2, -120.2),
        ("Portrait Photo", 35.0, -110.0),  # Different semantic cluster
        ("Person Photo", 35.1, -110.1),
        ("Family Photo", 35.2, -110.2),
        ("Food Photo", 25.0, -100.0),  # Another semantic cluster
        ("Meal Photo", 25.1, -100.1),
        ("Recipe Photo", 25.2, -100.2),
    ]

    for i, (name, clip_lat, clip_lon) in enumerate(locations):
        data.append(
            {
                SchemaColumns.UUID_STRING: f"clip_uuid_{i}",
                SchemaColumns.DATA_TYPE: "PHOTO",
                SchemaColumns.NAME: name,
                SchemaColumns.CAPTION: f"Photo {name}",
                SchemaColumns.SOURCE_FILE: f"clip_file_{i}.jpg",
                SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE: "2024-01-15T12:00:00",
                SchemaColumns.TIMESTAMP_UTC: datetime.datetime(
                    2024, 1, 15, 12, i, 0, tzinfo=datetime.timezone.utc
                ),
                SchemaColumns.GPS_LATITUDE: 40.0 + i * 0.1,  # Original GPS coordinates
                SchemaColumns.GPS_LONGITUDE: -74.0 + i * 0.1,
                SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE: clip_lat,  # CLIP embedding space
                SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE: clip_lon,
                SchemaColumns.RATING_0_TO_5: 4,
                SchemaColumns.SIZE_IN_BYTES: 1024000,
                SchemaColumns.COUNT: 1 + i * 2,  # Varying count values
            }
        )

    df = pd.DataFrame(data)
    df.index.name = SchemaColumns.DF_ID
    return df


def test_clip_embedding_plot_no_data(
    snapshot: SnapshotAssertion, individual_clip_data: pd.DataFrame
) -> None:
    fig = create_clip_embedding_plot(
        individual_clip_data.head(0),
        title="Test CLIP Embedding Plot - No Data",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_individual_data(
    snapshot: SnapshotAssertion, individual_clip_data: pd.DataFrame
) -> None:
    fig = create_clip_embedding_plot(
        individual_clip_data,
        title="Test CLIP Embedding Plot - Individual",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_individual_data_grouped(
    snapshot: SnapshotAssertion, individual_clip_data: pd.DataFrame
) -> None:
    fig = create_clip_embedding_plot(
        individual_clip_data,
        title="Test CLIP Embedding Plot - Individual, Grouped By Type",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_with_count_data(
    snapshot: SnapshotAssertion, grouped_clip_data: pd.DataFrame
) -> None:
    fig = create_clip_embedding_plot(
        grouped_clip_data,
        title="Test CLIP Embedding Plot - With Count Data",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_with_count_data_grouped(
    snapshot: SnapshotAssertion, grouped_clip_data: pd.DataFrame
) -> None:
    fig = create_clip_embedding_plot(
        grouped_clip_data,
        title="Test CLIP Embedding Plot - With Count Data, Grouped By Type",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_semantic_space_clustering(
    snapshot: SnapshotAssertion, clip_embedding_space_data: pd.DataFrame
) -> None:
    """Test that plot shows semantic clustering in CLIP embedding space."""
    fig = create_clip_embedding_plot(
        clip_embedding_space_data,
        title="Test CLIP Embedding Plot - Semantic Space Clustering",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_missing_coordinates(
    snapshot: SnapshotAssertion, individual_clip_data: pd.DataFrame
) -> None:
    """Test plot handles missing CLIP UMAP coordinates gracefully."""
    # Remove CLIP coordinates from some rows to test handling of missing data
    data_with_missing = individual_clip_data.copy()
    data_with_missing.loc[
        data_with_missing.index[:5], SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE
    ] = None
    data_with_missing.loc[
        data_with_missing.index[:5], SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE
    ] = None

    fig = create_clip_embedding_plot(
        data_with_missing,
        title="Test CLIP Embedding Plot - Missing Coordinates",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_clip_embedding_plot_no_clip_coordinates(
    snapshot: SnapshotAssertion, sample_data: pd.DataFrame
) -> None:
    """Test plot handles data without CLIP coordinates."""
    # Use original sample data without CLIP coordinates
    fig = create_clip_embedding_plot(
        sample_data,
        title="Test CLIP Embedding Plot - No CLIP Coordinates",
        clip_embedding_projection_state=ClipEmbeddingProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
