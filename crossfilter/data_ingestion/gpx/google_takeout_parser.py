"""Google Takeout location history parsing functions that convert JSON files to DataFrames."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from crossfilter.core.bucketing import add_geo_h3_bucket_columns
from crossfilter.core.schema import DataType
from crossfilter.core.schema import SchemaColumns as C

logger = logging.getLogger(__name__)


def generate_uuid_from_components(
    first_point_timestamp: datetime,
    first_point_lat: float,
    first_point_lon: float,
    point_timestamp: datetime,
    point_lat: float,
    point_lon: float,
    point_type: DataType,
) -> bytes:
    """Generate a deterministic UUID from the specified components."""
    # Create a string representation of all components
    components = [
        first_point_timestamp.isoformat(),
        str(first_point_lat),
        str(first_point_lon),
        point_timestamp.isoformat(),
        str(point_lat),
        str(point_lon),
        point_type.value,
    ]

    # Create a deterministic hash from the components
    component_string = "|".join(components)
    hash_digest = hashlib.sha256(component_string.encode("utf-8")).digest()
    return hash_digest[:16]


def generate_uuid_for_row(
    row: pd.Series,
    first_point_timestamp: datetime,
    first_point_lat: float,
    first_point_lon: float,
) -> bytes:
    """Generate a UUID for a single DataFrame row."""
    return generate_uuid_from_components(
        first_point_timestamp,
        first_point_lat,
        first_point_lon,
        row[C.TIMESTAMP_UTC].to_pydatetime(),
        row[C.GPS_LATITUDE],
        row[C.GPS_LONGITUDE],
        DataType.GOOGLE_TAKEOUT_LOCATION,
    )


def parse_takeout_timestamp(timestamp_str: str) -> datetime:
    """
    Parse Google Takeout timestamp string to datetime object.
    
    Google Takeout uses ISO 8601 format timestamps like '2022-01-12T17:18:24.190Z'
    """
    try:
        # Parse ISO 8601 timestamp
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        raise ValueError(f"Failed to parse timestamp '{timestamp_str}'") from e


def convert_e7_coordinate(e7_coord: int) -> float:
    """
    Convert Google's E7 coordinate format to decimal degrees.
    
    Google stores coordinates multiplied by 10^7 as integers.
    For example, latitudeE7: 414216106 becomes 41.4216106 degrees.
    """
    return e7_coord / 1e7


def parse_location_record(record: dict[str, Any], source_filename: str) -> dict[str, Any] | None:
    """
    Parse a single location record from Google Takeout Records.json.
    
    Returns None if the record is invalid or missing required fields.
    """
    # Check for required fields
    if "timestamp" not in record:
        logger.debug("Skipping record without timestamp")
        return None
        
    if "latitudeE7" not in record or "longitudeE7" not in record:
        logger.debug("Skipping record without coordinates")
        return None

    try:
        timestamp = parse_takeout_timestamp(record["timestamp"])
        latitude = convert_e7_coordinate(record["latitudeE7"])
        longitude = convert_e7_coordinate(record["longitudeE7"])
        
        # Validate coordinate ranges
        if not (-90 <= latitude <= 90):
            logger.warning(f"Invalid latitude {latitude}, skipping record")
            return None
        if not (-180 <= longitude <= 180):
            logger.warning(f"Invalid longitude {longitude}, skipping record")
            return None
        
        return {
            C.DATA_TYPE: DataType.GOOGLE_TAKEOUT_LOCATION,
            C.SOURCE_FILE: source_filename,
            C.TIMESTAMP_MAYBE_TIMEZONE_AWARE: record["timestamp"],
            C.TIMESTAMP_UTC: timestamp,
            C.GPS_LATITUDE: latitude,
            C.GPS_LONGITUDE: longitude,
        }
        
    except (ValueError, KeyError) as e:
        logger.debug(f"Failed to parse location record: {e}")
        return None


def load_google_takeout_records_to_df(records_json_path: Path) -> pd.DataFrame:
    """
    Parse a Google Takeout Records.json file and convert it to a DataFrame conforming to the schema.

    Args:
        records_json_path: Path to the Records.json file from Google Takeout

    Returns:
        DataFrame with location data as GPX trackpoints

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the JSON file is invalid or has unexpected structure
    """
    if not records_json_path.exists():
        raise FileNotFoundError(f"Google Takeout Records.json file not found: {records_json_path}")

    try:
        with open(records_json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {records_json_path}") from e
    except Exception as e:
        raise ValueError(f"Failed to read JSON file {records_json_path}") from e

    if not isinstance(data, dict) or "locations" not in data:
        raise ValueError(f"Invalid Google Takeout Records.json structure: missing 'locations' array in {records_json_path}")

    locations = data["locations"]
    if not isinstance(locations, list):
        raise ValueError(f"Invalid Google Takeout Records.json structure: 'locations' is not an array in {records_json_path}")

    # Collect all valid location records
    records: list[dict] = []
    source_filename = records_json_path.name

    logger.info(f"Processing {len(locations)} location records from {records_json_path}")
    
    for i, location in enumerate(locations):
        if not isinstance(location, dict):
            logger.debug(f"Skipping non-dict location record at index {i}")
            continue
            
        parsed_record = parse_location_record(location, source_filename)
        if parsed_record:
            records.append(parsed_record)

    if not records:
        logger.warning(f"No valid location records found in {records_json_path}")
        # Return empty DataFrame with correct schema
        df = pd.DataFrame(
            columns=[
                C.UUID,
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

    # Convert columns to proper types
    if not df.empty:
        df[C.TIMESTAMP_UTC] = pd.to_datetime(df[C.TIMESTAMP_UTC], utc=True)
        df[C.GPS_LATITUDE] = df[C.GPS_LATITUDE].astype(float)
        df[C.GPS_LONGITUDE] = df[C.GPS_LONGITUDE].astype(float)
        df = df.sort_values(
            by=[C.TIMESTAMP_UTC, C.DATA_TYPE, C.GPS_LATITUDE, C.GPS_LONGITUDE]
        )

        # Find the first point for UUID generation (first chronologically)
        first_row = df.iloc[0]
        first_point_timestamp = first_row[C.TIMESTAMP_UTC].to_pydatetime()
        first_point_lat = first_row[C.GPS_LATITUDE]
        first_point_lon = first_row[C.GPS_LONGITUDE]

        # Generate UUIDs for all rows using vectorized operation
        df[C.UUID] = df.apply(
            lambda row: generate_uuid_for_row(
                row, first_point_timestamp, first_point_lat, first_point_lon
            ),
            axis=1,
        )

    # Set index name
    df.index.name = C.DF_ID

    # Add H3 spatial index columns (parallelized computation)
    if not df.empty and C.GPS_LATITUDE in df.columns and C.GPS_LONGITUDE in df.columns:
        add_geo_h3_bucket_columns(df)
        logger.debug(f"Added H3 columns to {len(df)} points from {records_json_path}")

    logger.info(f"Loaded {len(df)} location points from {records_json_path}")

    return df