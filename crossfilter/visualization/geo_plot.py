"""Geo Plot Module.

Geographic scatter plot using Plotly tile scatter maps.
"""

import math
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from crossfilter.core.schema import SchemaColumns as C


def create_geo_plot(
    df: pd.DataFrame,
    title: str = "Geographic Distribution",
    groupby: Optional[str] = str(C.DATA_TYPE),
    max_marker_size: int = 50,
) -> go.Figure:
    """Create a Plotly geographic scatter plot with tile maps."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Check if we have required geographic columns
    required_geo_cols = [C.GPS_LATITUDE, C.GPS_LONGITUDE]
    missing_cols = [col for col in required_geo_cols if col not in df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing geographic columns: {', '.join(missing_cols)}",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Filter out rows with null coordinates
    df = df.dropna(subset=[C.GPS_LATITUDE, C.GPS_LONGITUDE]).copy()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid geographic coordinates",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df[C.DF_ID] = df.index
    if C.COUNT not in df.columns:
        df[C.COUNT] = 1

    # Handle groupby
    if not groupby:
        groupby = "Data"
        df[groupby] = "All"
    df[groupby] = df[groupby].astype(str).fillna("Unknown")
    df["groupby_count_sum"] = df.groupby(groupby)[C.COUNT].transform("sum")
    df["Group (Count)"] = df[groupby] + " (" + df["groupby_count_sum"].astype(str) + ")"
    df = df.sort_values(by=["groupby_count_sum", groupby], ascending=[False, True])

    # Calculate marker sizes based on COUNT, normalized so largest COUNT gets max_marker_size
    max_count = df[C.COUNT].max()
    min_marker_size = 5  # Minimum visible size
    
    if max_count > 0:
        # Make area proportional to COUNT: radius = sqrt(area) = sqrt(COUNT * scale_factor)
        # For largest COUNT, we want radius = max_marker_size
        # So: max_marker_size = sqrt(max_count * scale_factor)
        # Therefore: scale_factor = (max_marker_size^2) / max_count
        scale_factor = (max_marker_size ** 2) / max_count
        df["marker_size"] = df[C.COUNT].apply(
            lambda x: max(min_marker_size, math.sqrt(x * scale_factor))
        )
    else:
        df["marker_size"] = min_marker_size

    # Build hover_data list based on available columns
    hover_data_columns = [C.DF_ID, C.COUNT]
    for col in [C.NAME, C.DATA_TYPE, C.UUID_STRING, C.TIMESTAMP_UTC]:
        if col in df.columns:
            hover_data_columns.append(col)

    fig = px.scatter_map(
        df,
        lat=C.GPS_LATITUDE,
        lon=C.GPS_LONGITUDE,
        size="marker_size",
        color="Group (Count)",
        custom_data=[C.DF_ID],
        hover_data=hover_data_columns,
        title=title,
        map_style="open-street-map",
        size_max=max_marker_size,
    )

    # Configure layout
    fig.update_layout(
        hovermode="closest",
        showlegend=True,
        map=dict(
            center=dict(
                lat=df[C.GPS_LATITUDE].mean(),
                lon=df[C.GPS_LONGITUDE].mean(),
            ),
            zoom=10,
        ),
    )

    return fig