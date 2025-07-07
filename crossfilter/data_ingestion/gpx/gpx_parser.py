"""GPX file parsing functions that convert GPX files to DataFrames."""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import gpxpy
import pandas as pd
from gpxpy.gpx import GPXTrackPoint, GPXWaypoint

from crossfilter.core.bucketing import add_geo_h3_bucket_columns
from crossfilter.core.schema import DataType, SchemaColumns as C

logger = logging.getLogger(__name__)


def generate_uuid_from_components(
    first_trackpoint_timestamp: datetime,
    first_trackpoint_lat: float,
    first_trackpoint_lon: float,
    point_timestamp: datetime,
    point_lat: float,
    point_lon: float,
    point_type: DataType,
) -> str:
    """Generate a deterministic UUID from the specified components."""
    # Create a string representation of all components
    components = [
        first_trackpoint_timestamp.isoformat(),
        str(first_trackpoint_lat),
        str(first_trackpoint_lon),
        point_timestamp.isoformat(),
        str(point_lat),
        str(point_lon),
        point_type.value,
    ]

    # Create a deterministic hash from the components
    component_string = "|".join(components)
    hash_digest = hashlib.sha256(component_string.encode("utf-8")).digest()

    # Create a UUID from the hash
    return str(uuid.UUID(bytes=hash_digest[:16]))


def generate_uuid_for_row(
    row: pd.Series,
    first_point_timestamp: datetime,
    first_point_lat: float,
    first_point_lon: float,
) -> str:
    """Generate a UUID for a single DataFrame row."""
    return generate_uuid_from_components(
        first_point_timestamp,
        first_point_lat,
        first_point_lon,
        row[C.TIMESTAMP_UTC].to_pydatetime(),
        row[C.GPS_LATITUDE],
        row[C.GPS_LONGITUDE],
        DataType(row[C.DATA_TYPE]),
    )


def load_gpx_file_to_df(gpx_file_path: Path) -> pd.DataFrame:
    """
    Parse a GPX file and convert it to a DataFrame conforming to the schema.

    Args:
        gpx_file_path: Path to the GPX file

    Returns:
        DataFrame with GPX trackpoints and waypoints

    Raises:
        FileNotFoundError: If the GPX file doesn't exist
        ValueError: If the GPX file is invalid or empty
    """
    if not gpx_file_path.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_file_path}")

    try:
        with open(gpx_file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except Exception as e:
        raise ValueError(f"Failed to parse GPX file {gpx_file_path}: {e}")

    # Get file size
    file_size = gpx_file_path.stat().st_size

    # Collect all points first (without UUIDs)
    records: List[dict] = []

    # Process all trackpoints
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time and point.latitude and point.longitude:
                    records.append(
                        {
                            C.DATA_TYPE: DataType.GPX_TRACKPOINT,
                            C.SOURCE_FILE: str(gpx_file_path.name),
                            C.TIMESTAMP_MAYBE_TIMEZONE_AWARE: point.time.isoformat(),
                            C.TIMESTAMP_UTC: (
                                point.time
                                if point.time.tzinfo is not None
                                else point.time.replace(tzinfo=timezone.utc)
                            ),
                            C.GPS_LATITUDE: point.latitude,
                            C.GPS_LONGITUDE: point.longitude,
                        }
                    )

    # Process all waypoints
    for waypoint in gpx.waypoints:
        if waypoint.time and waypoint.latitude and waypoint.longitude:
            records.append(
                {
                    C.DATA_TYPE: DataType.GPX_WAYPOINT,
                    C.NAME: waypoint.name,
                    C.CAPTION: waypoint.comment or waypoint.description,
                    C.SOURCE_FILE: str(gpx_file_path.name),
                    C.TIMESTAMP_MAYBE_TIMEZONE_AWARE: waypoint.time.isoformat(),
                    C.TIMESTAMP_UTC: (
                        waypoint.time
                        if waypoint.time.tzinfo is not None
                        else waypoint.time.replace(tzinfo=timezone.utc)
                    ),
                    C.GPS_LATITUDE: waypoint.latitude,
                    C.GPS_LONGITUDE: waypoint.longitude,
                }
            )

    if not records:
        logger.warning(f"No valid trackpoints or waypoints found in {gpx_file_path}")
        # Return empty DataFrame with correct schema
        df = pd.DataFrame(
            columns=[
                C.UUID_STRING,
                C.DATA_TYPE,
                C.NAME,
                C.SOURCE_FILE,
                C.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
                C.TIMESTAMP_UTC,
                C.GPS_LATITUDE,
                C.GPS_LONGITUDE,
            ]
        )
    else:
        # Create DataFrame
        df = pd.DataFrame(records)

    # Convert TIMESTAMP_UTC to proper datetime with UTC timezone
    df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
    df[C.GPS_LATITUDE] = df[C.GPS_LATITUDE].astype(float)
    df[C.GPS_LONGITUDE] = df[C.GPS_LONGITUDE].astype(float)
    df = df.sort_values(
        by=[C.TIMESTAMP_UTC, C.DATA_TYPE, C.GPS_LATITUDE, C.GPS_LONGITUDE]
    )

    if not df.empty:
        # Find the first point for UUID generation (first chronologically)
        first_row = df.iloc[0]
        first_point_timestamp = first_row[C.TIMESTAMP_UTC].to_pydatetime()
        first_point_lat = first_row[C.GPS_LATITUDE]
        first_point_lon = first_row[C.GPS_LONGITUDE]

        # Generate UUIDs for all rows using vectorized operation
        df[C.UUID_STRING] = df.apply(
            lambda row: generate_uuid_for_row(
                row, first_point_timestamp, first_point_lat, first_point_lon
            ),
            axis=1,
        )

    # Set index name
    df.index.name = C.DF_ID

    # Add H3 spatial index columns for each individual GPX file (parallelized computation)
    if not df.empty and C.GPS_LATITUDE in df.columns and C.GPS_LONGITUDE in df.columns:
        add_geo_h3_bucket_columns(df)
        logger.debug(f"Added H3 columns to {len(df)} points from {gpx_file_path}")

    logger.info(f"Loaded {len(df)} points from {gpx_file_path}")

    return df
