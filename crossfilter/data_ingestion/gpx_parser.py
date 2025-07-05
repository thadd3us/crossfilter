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

from crossfilter.core.schema import DataType, SchemaColumns

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
        row[SchemaColumns.TIMESTAMP_UTC].to_pydatetime().replace(tzinfo=timezone.utc),
        row[SchemaColumns.GPS_LATITUDE],
        row[SchemaColumns.GPS_LONGITUDE],
        DataType(row[SchemaColumns.DATA_TYPE]),
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
        track_name = track.name or None
        for segment in track.segments:
            for point in segment.points:
                if point.time and point.latitude and point.longitude:
                    records.append({
                        SchemaColumns.DATA_TYPE: DataType.GPX_TRACKPOINT,
                        SchemaColumns.NAME: track_name,
                        SchemaColumns.CAPTION: None,
                        SchemaColumns.SOURCE_FILE: str(gpx_file_path),
                        SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE: point.time.isoformat(),
                        SchemaColumns.TIMESTAMP_UTC: point.time if point.time.tzinfo is not None else point.time.replace(tzinfo=timezone.utc),
                        SchemaColumns.GPS_LATITUDE: point.latitude,
                        SchemaColumns.GPS_LONGITUDE: point.longitude,
                        SchemaColumns.RATING_0_TO_5: None,
                        SchemaColumns.SIZE_IN_BYTES: file_size,
                    })
    
    # Process all waypoints
    for waypoint in gpx.waypoints:
        if waypoint.time and waypoint.latitude and waypoint.longitude:
            records.append({
                SchemaColumns.DATA_TYPE: DataType.GPX_WAYPOINT,
                SchemaColumns.NAME: waypoint.name,
                SchemaColumns.CAPTION: waypoint.comment or waypoint.description,
                SchemaColumns.SOURCE_FILE: str(gpx_file_path),
                SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE: waypoint.time.isoformat(),
                SchemaColumns.TIMESTAMP_UTC: waypoint.time if waypoint.time.tzinfo is not None else waypoint.time.replace(tzinfo=timezone.utc),
                SchemaColumns.GPS_LATITUDE: waypoint.latitude,
                SchemaColumns.GPS_LONGITUDE: waypoint.longitude,
                SchemaColumns.RATING_0_TO_5: None,
                SchemaColumns.SIZE_IN_BYTES: file_size,
            })
    
    if not records:
        logger.warning(f"No valid trackpoints or waypoints found in {gpx_file_path}")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            SchemaColumns.UUID_STRING,
            SchemaColumns.DATA_TYPE,
            SchemaColumns.NAME,
            SchemaColumns.CAPTION,
            SchemaColumns.SOURCE_FILE,
            SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
            SchemaColumns.TIMESTAMP_UTC,
            SchemaColumns.GPS_LATITUDE,
            SchemaColumns.GPS_LONGITUDE,
            SchemaColumns.RATING_0_TO_5,
            SchemaColumns.SIZE_IN_BYTES,
        ])
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Convert TIMESTAMP_UTC to proper datetime with UTC timezone
    df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(df[SchemaColumns.TIMESTAMP_UTC], utc=True)
    
    # Find the first point for UUID generation (first chronologically)
    first_row = df.loc[df[SchemaColumns.TIMESTAMP_UTC].idxmin()]
    first_point_timestamp = first_row[SchemaColumns.TIMESTAMP_UTC].to_pydatetime().replace(tzinfo=timezone.utc)
    first_point_lat = first_row[SchemaColumns.GPS_LATITUDE]
    first_point_lon = first_row[SchemaColumns.GPS_LONGITUDE]
    
    # Generate UUIDs for all rows using vectorized operation
    df[SchemaColumns.UUID_STRING] = df.apply(
        lambda row: generate_uuid_for_row(row, first_point_timestamp, first_point_lat, first_point_lon),
        axis=1
    )
    
    # Set index name
    df.index.name = SchemaColumns.DF_ID
    
    logger.info(f"Loaded {len(df)} points from {gpx_file_path}")
    
    return df