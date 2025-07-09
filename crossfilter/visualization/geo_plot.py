"""Geo Plot Module.

Geographic scatter plot using Plotly tile scatter maps.
"""

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import SchemaColumns as C
from crossfilter.visualization.plot_common import CUSTOM_DATA_COLUMNS
from crossfilter.core.bucketing import groupby_fillna


def _haversine_distance_vectorized(
    lat1: float, lon1: float, lat2_array: np.ndarray, lon2_array: np.ndarray
) -> np.ndarray:
    """Calculate the great circle distance between one point and an array of points using the haversine formula.

    Args:
        lat1, lon1: Single point coordinates in degrees
        lat2_array, lon2_array: Array of coordinates in degrees

    Returns:
        Array of distances in meters
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)

    # Haversine formula (vectorized)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters
    R = 6371000
    distances = R * c

    return distances


def _calculate_geographic_center_and_radius(
    latitudes: pd.Series, longitudes: pd.Series
) -> Tuple[float, float, float]:
    """Calculate the geographic center and radius in meters using proper spherical geometry.

    Returns:
        Tuple of (center_lat, center_lon, radius_meters)
    """
    # Use the spherical geometry approach for center calculation to handle longitude wrapping
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

    # Calculate maximum distance from center to any point using vectorized haversine formula
    distances = _haversine_distance_vectorized(
        center_lat, center_lon, latitudes.values, longitudes.values
    )
    max_distance = np.max(distances)

    return center_lat, center_lon, max_distance


def _calculate_zoom_level_from_radius(
    radius_meters: float, center_lat: float, plot_size_pixels: int = 400
) -> int:
    """Calculate zoom level based on radius in meters and plot size.

    Uses Mapbox zoom level formula: https://docs.mapbox.com/help/glossary/zoom-level/
    At zoom level 0: 1 pixel = ~156,543 meters at the equator
    At zoom level z: 1 pixel = 156,543 / (2^z) meters at the equator

    This needs to be adjusted for latitude using cos(latitude).

    Args:
        radius_meters: Radius of the circle containing all points in meters
        center_lat: Center latitude for projection adjustment
        plot_size_pixels: Size of the plot in pixels (assumed square)

    Returns:
        Appropriate zoom level (0-20)
    """
    # Mapbox constants
    EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 = 156543.03392804097

    # We need the viewport to show a circle of radius_meters
    # So we need plot_size_pixels/2 pixels to cover radius_meters
    viewport_radius_pixels = plot_size_pixels / 2

    # Calculate required meters per pixel
    required_meters_per_pixel = radius_meters / viewport_radius_pixels

    # Adjust for latitude (be conservative - use the latitude that gives the most zoomed out view)
    # At higher latitudes, the same degree of longitude covers fewer meters
    # So we use cos(lat) to adjust, but we want the most conservative (zoomed out) view
    # The most conservative latitude is the one closest to the equator (smallest absolute value)
    lat_adjustment = math.cos(math.radians(abs(center_lat)))

    # Adjust the required meters per pixel for latitude
    adjusted_meters_per_pixel = required_meters_per_pixel / lat_adjustment

    # Calculate zoom level: zoom = log2(EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 / adjusted_meters_per_pixel)
    if adjusted_meters_per_pixel <= 0:
        return 20  # Maximum zoom for very small areas

    zoom = math.log2(EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 / adjusted_meters_per_pixel)

    # Clamp to reasonable zoom levels
    zoom = max(0, min(20, int(zoom)))

    return zoom


def _calculate_map_view(
    latitudes: pd.Series, longitudes: pd.Series
) -> Tuple[float, float, int]:
    """Calculate the optimal map center and zoom level for the given geographic points."""
    if len(latitudes) == 0:
        return 0.0, 0.0, 1

    if len(latitudes) == 1:
        # Single point - use high zoom
        return latitudes.iloc[0], longitudes.iloc[0], 14

    # Calculate geographic center and radius using GeoPandas
    center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
        latitudes, longitudes
    )

    # Calculate zoom level based on radius
    zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)

    return center_lat, center_lon, zoom


def create_geo_plot(
    df: pd.DataFrame,
    geo_projection_state: GeoProjectionState,
    title: Optional[str] = None,
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
    groupby = geo_projection_state.projection_state.groupby_column
    if not groupby:
        groupby = "Data"
        df[groupby] = "All"
    assert groupby in df.columns, f"Groupby column {groupby} not found in DataFrame"
    df[groupby] = groupby_fillna(df[groupby])

    df["groupby_count_sum"] = df.groupby(groupby)[C.COUNT].transform("sum")
    df["Group (Count)"] = df[groupby] + " (" + df["groupby_count_sum"].astype(str) + ")"
    df = df.sort_values(by=["groupby_count_sum", groupby], ascending=[False, True])

    # Calculate marker sizes based on COUNT, normalized so largest COUNT gets max_marker_size
    max_count = df[C.COUNT].max()
    min_marker_size = 3  # Minimum visible size

    if max_count > 0:
        # Make area proportional to COUNT: radius = sqrt(area) = sqrt(COUNT * scale_factor)
        # For largest COUNT, we want radius = max_marker_size
        # So: max_marker_size = sqrt(max_count * scale_factor)
        # Therefore: scale_factor = (max_marker_size^2) / max_count
        scale_factor = (geo_projection_state.max_marker_size**2) / max_count
        df["marker_size"] = df[C.COUNT].apply(
            lambda x: max(min_marker_size, math.sqrt(x * scale_factor))
        )
    else:
        df["marker_size"] = min_marker_size

    fig = px.scatter_map(
        df,
        lat=C.GPS_LATITUDE,
        lon=C.GPS_LONGITUDE,
        size="marker_size",
        color="Group (Count)",
        custom_data=CUSTOM_DATA_COLUMNS,
        hover_data={
            C.DF_ID: True,
            C.COUNT: True,
            C.UUID_STRING: True,
        },
        title=title,
        map_style="open-street-map",
        size_max=geo_projection_state.max_marker_size,
    )

    # Calculate proper geographic bounds and center
    center_lat, center_lon, zoom = _calculate_map_view(
        df[C.GPS_LATITUDE], df[C.GPS_LONGITUDE]
    )

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
