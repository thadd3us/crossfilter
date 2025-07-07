"""CLIP Embedding Plot Module.

CLIP embedding visualization using the same geographic scatter plot logic as geo_plot
but with CLIP UMAP coordinates instead of GPS coordinates.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from crossfilter.core.clip_embedding_projection_state import (
    ClipEmbeddingProjectionState,
)
from crossfilter.core.schema import SchemaColumns as C
from crossfilter.visualization.geo_plot import create_geo_plot


def create_clip_embedding_plot(
    df: pd.DataFrame,
    clip_embedding_projection_state: ClipEmbeddingProjectionState,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a Plotly CLIP embedding scatter plot using geographic plot logic.

    This function reuses the create_geo_plot function but with CLIP UMAP coordinates
    instead of GPS coordinates. The CLIP embeddings have been projected into a 2D
    latitude/longitude space that has an identical span as normal terrestrial lat/longs.

    Args:
        df: DataFrame containing CLIP embedding data
        clip_embedding_projection_state: State object for CLIP embedding projection
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    # Create a temporary copy of the dataframe with CLIP coordinates mapped to GPS columns
    df_temp = df.copy()

    # Map CLIP UMAP coordinates to GPS columns for reuse of geo_plot logic
    if C.CLIP_UMAP_HAVERSINE_LATITUDE in df_temp.columns:
        df_temp[C.GPS_LATITUDE] = df_temp[C.CLIP_UMAP_HAVERSINE_LATITUDE]
    if C.CLIP_UMAP_HAVERSINE_LONGITUDE in df_temp.columns:
        df_temp[C.GPS_LONGITUDE] = df_temp[C.CLIP_UMAP_HAVERSINE_LONGITUDE]

    # Create a temporary geo projection state that matches the clip embedding state
    from crossfilter.core.geo_projection_state import GeoProjectionState

    temp_geo_state = GeoProjectionState(
        clip_embedding_projection_state.projection_state.max_rows
    )
    temp_geo_state.projection_state = clip_embedding_projection_state.projection_state
    temp_geo_state.current_h3_level = clip_embedding_projection_state.current_h3_level
    temp_geo_state.max_marker_size = clip_embedding_projection_state.max_marker_size

    # Use the existing geo_plot function with our temporary state
    fig = create_geo_plot(df_temp, temp_geo_state, title)

    return fig
