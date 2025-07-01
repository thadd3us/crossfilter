"""Visualization components for geographic and temporal crossfilter plots."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_kepler_config(df: pd.DataFrame) -> dict[str, Any]:
        """
        Create Kepler.gl configuration for geographic heatmap.

        Args:
            df: DataFrame with geographic data

        Returns:
            Kepler.gl configuration dictionary
        """
        if df.empty:
            return {
                "version": "v1",
                "config": {
                    "visState": {
                        "layers": [],
                        "filters": [],
                        "interactionConfig": {
                            "tooltip": {"enabled": True},
                            "brush": {"enabled": True}
                        }
                    },
                    "mapState": {
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "zoom": 10
                    }
                }
            }

        # Determine if we have aggregated or individual points
        has_count_col = 'count' in df.columns

        # Create layer configuration
        if has_count_col:
            # Heatmap layer for aggregated data
            layer_config = {
                "id": "heatmap_layer",
                "type": "heatmap",
                "config": {
                    "dataId": "crossfilter_data",
                    "label": "Geographic Heatmap",
                    "color": [255, 153, 31],
                    "columns": {
                        "lat": "lat",
                        "lng": "lon"
                    },
                    "isVisible": True,
                    "visConfig": {
                        "opacity": 0.8,
                        "colorRange": {
                            "name": "Global Warming",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": [
                                "#5A1846", "#900C3F", "#C70039",
                                "#E3611C", "#F1920E", "#FFC300"
                            ]
                        },
                        "radius": 20
                    }
                },
                "visualChannels": {
                    "weightField": {
                        "name": "count",
                        "type": "integer"
                    }
                }
            }
        else:
            # Point layer for individual data points
            layer_config = {
                "id": "point_layer",
                "type": "point",
                "config": {
                    "dataId": "crossfilter_data",
                    "label": "Data Points",
                    "color": [23, 184, 190],
                    "columns": {
                        "lat": "GPS_LATITUDE",
                        "lng": "GPS_LONGITUDE"
                    },
                    "isVisible": True,
                    "visConfig": {
                        "opacity": 0.8,
                        "strokeOpacity": 0.8,
                        "thickness": 2,
                        "strokeColor": [255, 255, 255],
                        "colorRange": {
                            "name": "Global Warming",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": [
                                "#5A1846", "#900C3F", "#C70039",
                                "#E3611C", "#F1920E", "#FFC300"
                            ]
                        },
                        "strokeColorRange": {
                            "name": "Global Warming",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": [
                                "#5A1846", "#900C3F", "#C70039",
                                "#E3611C", "#F1920E", "#FFC300"
                            ]
                        },
                        "radius": 3,
                        "sizeRange": [1, 10],
                        "radiusRange": [1, 50]
                    }
                }
            }

        # Calculate map bounds
        if has_count_col:
            lat_col, lon_col = 'lat', 'lon'
        else:
            lat_col, lon_col = 'GPS_LATITUDE', 'GPS_LONGITUDE'

        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()

        # Estimate zoom level based on data spread
        lat_range = df[lat_col].max() - df[lat_col].min()
        lon_range = df[lon_col].max() - df[lon_col].min()
        max_range = max(lat_range, lon_range)

        if max_range > 10:
            zoom = 3
        elif max_range > 1:
            zoom = 6
        elif max_range > 0.1:
            zoom = 9
        else:
            zoom = 12

        config = {
            "version": "v1",
            "config": {
                "visState": {
                    "layers": [layer_config],
                    "filters": [],
                    "interactionConfig": {
                        "tooltip": {
                            "enabled": True,
                            "fieldsToShow": {
                                "crossfilter_data": [
                                    {"name": "UUID_LONG", "format": None} if not has_count_col else {"name": "count", "format": None}
                                ]
                            }
                        },
                        "brush": {"enabled": True, "size": 0.5},
                        "geocoder": {"enabled": False}
                    }
                },
                "mapState": {
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "zoom": zoom,
                    "pitch": 0,
                    "bearing": 0,
                    "dragRotate": False,
                    "isSplit": False
                },
                "mapStyle": {
                    "styleType": "dark",
                    "topLayerGroups": {},
                    "visibleLayerGroups": {
                        "label": True,
                        "road": True,
                        "border": False,
                        "building": True,
                        "water": True,
                        "land": True,
                        "3d building": False
                    },
                    "threeDBuildingColor": [9.665468314072013, 17.18305478057247, 31.1442867897876],
                    "mapStyles": {}
                }
            }
        }

        return config


def prepare_kepler_data(df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Prepare data for Kepler.gl visualization.

        Args:
            df: DataFrame with geographic data

        Returns:
            List of data rows for Kepler.gl
        """
        if df.empty:
            return []

        # Convert DataFrame to list of dictionaries
        return df.to_dict('records')


def create_fallback_scatter_geo(df: pd.DataFrame, title: str = "Geographic Distribution") -> dict[str, Any]:
        """
        Create a fallback Plotly scatter_geo plot if Kepler.gl is not available.

        Args:
            df: DataFrame with geographic data
            title: Plot title

        Returns:
            Plotly figure as dictionary
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            # Determine column names
            if 'count' in df.columns:
                # Aggregated data
                lat_col, lon_col = 'lat', 'lon'
                size_col = 'count'
                hover_data = ['count']
            else:
                # Individual points
                lat_col, lon_col = 'GPS_LATITUDE', 'GPS_LONGITUDE'
                size_col = None
                # Use df_id if available, otherwise no hover data
                hover_data = ['df_id'] if 'df_id' in df.columns else None

            fig = px.scatter_geo(
                df,
                lat=lat_col,
                lon=lon_col,
                size=size_col,
                hover_data=hover_data,
                title=title
            )

            fig.update_geos(
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                coastlinecolor="rgb(204, 204, 204)"
            )

        fig.update_layout(
            height=500,
            margin={"l": 0, "r": 0, "t": 50, "b": 0}
        )

        return fig.to_dict()
