"""Geo Plot Module.

Geographic scatter plot using Plotly tile scatter maps.
"""

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from crossfilter.core.schema import SchemaColumns as C


def _calculate_geographic_center(latitudes: pd.Series, longitudes: pd.Series) -> Tuple[float, float]:
    """Calculate the geographic center of a set of points, handling longitude wrapping."""
    # Convert to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    
    # Convert to Cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    # Calculate mean in Cartesian space
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    
    # Convert back to spherical coordinates
    center_lat = np.degrees(np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2)))
    center_lon = np.degrees(np.arctan2(y_mean, x_mean))
    
    return center_lat, center_lon


def _calculate_geographic_bounds(latitudes: pd.Series, longitudes: pd.Series) -> Tuple[float, float, float, float]:
    """Calculate geographic bounds. For most use cases, simple min/max works fine."""
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_min, lon_max = longitudes.min(), longitudes.max()
    
    # Simple bounds - let Plotly handle the complex projection issues
    return lat_min, lat_max, lon_min, lon_max


def _calculate_zoom_level(lat_span: float, lon_span: float, center_lat: float) -> int:
    """Calculate appropriate zoom level based on geographic span and latitude."""
    # Account for Mercator projection distortion - longitude degrees get "wider" near the poles
    # Adjust longitude span by the cosine of the latitude
    adjusted_lon_span = lon_span * np.cos(np.radians(center_lat))
    
    # Use the larger of the two spans (latitude or adjusted longitude)
    effective_span = max(lat_span, adjusted_lon_span)
    
    # Zoom level calculation based on effective span
    # These thresholds are tuned for good visual results
    if effective_span > 120:  # Global view
        zoom = 1
    elif effective_span > 60:  # Continental view
        zoom = 2
    elif effective_span > 30:  # Large country/region
        zoom = 3
    elif effective_span > 15:  # Country view
        zoom = 4
    elif effective_span > 8:  # State/province view
        zoom = 5
    elif effective_span > 4:  # Regional view
        zoom = 6
    elif effective_span > 2:  # Metropolitan area
        zoom = 7
    elif effective_span > 1:  # City view
        zoom = 8
    elif effective_span > 0.5:  # District view
        zoom = 9
    elif effective_span > 0.25:  # Neighborhood view
        zoom = 10
    elif effective_span > 0.1:  # Local area
        zoom = 11
    elif effective_span > 0.05:  # Street level
        zoom = 12
    elif effective_span > 0.01:  # Block level
        zoom = 13
    else:  # Very local/building level
        zoom = 14
    
    return zoom


def _calculate_map_view(latitudes: pd.Series, longitudes: pd.Series) -> Tuple[float, float, int]:
    """Calculate the optimal map center and zoom level for the given geographic points."""
    # Calculate geographic center using proper spherical geometry
    center_lat, center_lon = _calculate_geographic_center(latitudes, longitudes)
    
    # Calculate proper bounds
    lat_min, lat_max, lon_min, lon_max = _calculate_geographic_bounds(latitudes, longitudes)
    
    # Calculate spans
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    
    # Handle longitude wrapping for span calculation
    if lon_span > 180:
        # We're crossing the date line, so the actual span is smaller
        lon_span = 360 - lon_span
    
    # Calculate zoom level
    zoom = _calculate_zoom_level(lat_span, lon_span, center_lat)
    
    return center_lat, center_lon, zoom


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

    # Calculate proper geographic bounds and center
    center_lat, center_lon, zoom = _calculate_map_view(df[C.GPS_LATITUDE], df[C.GPS_LONGITUDE])

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
