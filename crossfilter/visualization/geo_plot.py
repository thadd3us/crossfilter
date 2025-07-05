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
    title: Optional[str] = None,
    groupby: Optional[str] = str(C.DATA_TYPE),
    max_marker_size: int = 10,
) -> go.Figure:
    """Create a Plotly geographic scatter plot with tile maps."""
    df = df.dropna(subset=[C.GPS_LATITUDE, C.GPS_LONGITUDE]).copy()

    if df.empty:
        # Show world map when no data - create empty map manually
        fig = go.Figure()
        fig.add_trace(
            go.Scattermap(
                lat=[],
                lon=[],
                mode="markers",
            )
        )
        fig.update_layout(
            title=title,
            map=dict(
                style="open-street-map",
                center=dict(lat=0, lon=0),  # Center on equator
                zoom=1,  # World view
            ),
            annotations=[
                dict(
                    text="No data to display",
                    x=0.5,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    # Check if we have required geographic columns

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
    min_marker_size = 1  # Minimum visible size

    if max_count > 0:
        # Make area proportional to COUNT: radius = sqrt(area) = sqrt(COUNT * scale_factor)
        # For largest COUNT, we want radius = max_marker_size
        # So: max_marker_size = sqrt(max_count * scale_factor)
        # Therefore: scale_factor = (max_marker_size^2) / max_count
        scale_factor = (max_marker_size**2) / max_count
        df["marker_size"] = df[C.COUNT].apply(
            lambda x: max(min_marker_size, math.sqrt(x * scale_factor))
        )
    else:
        df["marker_size"] = min_marker_size

    # Build hover_data list based on available columns
    hover_data_columns = [C.COUNT]
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

    # Calculate bounds for auto-fitting the map
    lat_min, lat_max = df[C.GPS_LATITUDE].min(), df[C.GPS_LATITUDE].max()
    lon_min, lon_max = df[C.GPS_LONGITUDE].min(), df[C.GPS_LONGITUDE].max()

    # Calculate center
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Calculate zoom level based on data span
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    max_span = max(lat_span, lon_span)

    # Heuristic zoom calculation (adjust as needed)
    if max_span > 60:  # Very wide spread
        zoom = 2
    elif max_span > 20:  # Continental scale
        zoom = 4
    elif max_span > 5:  # Regional scale
        zoom = 6
    elif max_span > 1:  # City scale
        zoom = 8
    elif max_span > 0.1:  # Neighborhood scale
        zoom = 10
    else:  # Very local
        zoom = 12

    # Configure layout with auto-fitted bounds
    fig.update_layout(
        hovermode="closest",
        showlegend=True,
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
    )

    return fig
