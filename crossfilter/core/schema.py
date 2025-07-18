"""Schema definitions, constants, and enums for crossfilter data structures."""

import json
import logging
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# Import shared types from backend_frontend_shared_schema
from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    ProjectionType,
    TemporalLevel,
)

logger = logging.getLogger(__name__)


class SchemaColumns(StrEnum):
    """Column names from the DataSchema."""

    # Used to refer to the DataFrame integer index.
    DF_ID = "df_id"

    UUID_STRING = "UUID_STRING"
    DATA_TYPE = "DATA_TYPE"
    NAME = "NAME"
    CAPTION = "CAPTION"
    SOURCE_FILE = "SOURCE_FILE"
    TIMESTAMP_MAYBE_TIMEZONE_AWARE = "TIMESTAMP_MAYBE_TIMEZONE_AWARE"
    TIMESTAMP_UTC = "TIMESTAMP_UTC"
    GPS_LATITUDE = "GPS_LATITUDE"
    GPS_LONGITUDE = "GPS_LONGITUDE"
    RATING_0_TO_5 = "RATING_0_TO_5"
    SIZE_IN_BYTES = "SIZE_IN_BYTES"
    COUNT = "COUNT"

    # CLIP embedding UMAP projection coordinates (on a sphere like lat/lon)
    CLIP_UMAP_HAVERSINE_LATITUDE = "CLIP_UMAP_HAVERSINE_LATITUDE"
    CLIP_UMAP_HAVERSINE_LONGITUDE = "CLIP_UMAP_HAVERSINE_LONGITUDE"
    
    # SigLIP2 embedding UMAP projection coordinates (on a sphere like lat/lon)
    SIGLIP2_UMAP2D_HAVERSINE_LATITUDE = "SIGLIP2_UMAP2D_HAVERSINE_LATITUDE"
    SIGLIP2_UMAP2D_HAVERSINE_LONGITUDE = "SIGLIP2_UMAP2D_HAVERSINE_LONGITUDE"


class DataType(StrEnum):
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    GPX_TRACKPOINT = "GPX_TRACKPOINT"
    GPX_WAYPOINT = "GPX_WAYPOINT"


class EmbeddingType(StrEnum):
    SIGLIP2 = "SIGLIP2"

class DataSchema(pa.DataFrameModel):
    UUID_STRING: Series[str] = pa.Field(nullable=True)
    DATA_TYPE: Series[str] = pa.Field(
        nullable=True, isin=list(DataType.__members__.values())
    )
    NAME: Series[str] = pa.Field(nullable=True)
    CAPTION: Series[str] = pa.Field(nullable=True)
    SOURCE_FILE: Series[str] = pa.Field(nullable=True)
    TIMESTAMP_MAYBE_TIMEZONE_AWARE: Series[str] = pa.Field(nullable=True)
    TIMESTAMP_UTC: Series[pd.DatetimeTZDtype] = pa.Field(
        nullable=True, dtype_kwargs={"tz": "UTC"}
    )
    GPS_LATITUDE: Series[float] = pa.Field(nullable=True, ge=-90, le=90)
    GPS_LONGITUDE: Series[float] = pa.Field(nullable=True, ge=-180, le=180)
    RATING_0_TO_5: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0, le=5)
    SIZE_IN_BYTES: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0)

    # For aggregated data, the COUNT column says how many rows are aggregated into the bucket.
    # Some of the other columns may still be present, coming from a single row sample of the data aggregated into the bucket.
    # For single point data, COUNT is missing and implied to be 1.
    COUNT: Optional[Series[pd.Int64Dtype]]  # = pa.Field(nullable=True, ge=0)

    class Config:
        strict = True
        coerce = True


def get_h3_column_name(level: int) -> str:
    """Get the H3 column name for a specific level (0-15)."""
    if not 0 <= level <= 15:
        raise ValueError(f"H3 level must be between 0 and 15, got {level}")
    return f"BUCKETED_H3_L{level}"


def get_clip_umap_h3_column_name(level: int) -> str:
    """Get the CLIP UMAP H3 column name for a specific level (0-15)."""
    if not 0 <= level <= 15:
        raise ValueError(f"H3 level must be between 0 and 15, got {level}")
    return f"BUCKETED_CLIP_HAVERSINE_UMAP_H3_L{level}"


def get_temporal_column_name(level: TemporalLevel) -> str:
    """Get the temporal column name for a specific temporal level."""
    return f"BUCKETED_TIMESTAMP_{level}"


required_columns = [
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
]


def load_jsonl_to_dataframe(jsonl_path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a DataFrame conforming to DataSchema.

    The JSONL file may have extra columns and may not specify all required columns.
    Missing schema columns will be created with null values, and data will be coerced to
    the correct types according to the schema. Extra columns are preserved in the output.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        DataFrame with schema columns validated, plus any extra columns from JSONL
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    # Read JSONL file
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    df = pd.DataFrame(records)

    # Ensure all required columns exist, adding missing ones with null values

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Convert SOURCE_FILE from Path objects to strings, handling missing values
    df[SchemaColumns.SOURCE_FILE] = df[SchemaColumns.SOURCE_FILE].astype(str)

    # Convert TIMESTAMP_UTC to UTC timezone-aware datetime, handling missing values
    df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(
        df[SchemaColumns.TIMESTAMP_UTC], utc=True, errors="coerce"
    )

    # Validate and coerce only the schema columns
    schema_df = df[required_columns].copy()
    validated_schema_df = DataSchema.validate(schema_df, lazy=True)

    # Combine validated schema columns with any extra columns from original data
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns:
        result_df = pd.concat([validated_schema_df, df[extra_columns]], axis=1)
    else:
        result_df = validated_schema_df

    # Set stable df_id index using pandas int64 index
    result_df = result_df.reset_index(drop=True)
    result_df.index.name = SchemaColumns.DF_ID

    logger.info(f"Loaded {len(result_df)} records from {jsonl_path=}")

    return result_df


def load_sqlite_to_dataframe(sqlite_db_path: Path, table_name: str) -> pd.DataFrame:
    if not sqlite_db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_db_path}")

    # Read data from SQLite table
    with sqlite3.connect(sqlite_db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Convert TIMESTAMP_UTC to UTC timezone-aware datetime, handling missing values
    # THAD: FIXME -- why is this needed?
    df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(
        df[SchemaColumns.TIMESTAMP_UTC], utc=True, errors="coerce"
    )
    df[SchemaColumns.GPS_LATITUDE] = df[SchemaColumns.GPS_LATITUDE].clip(-90, 90)
    df[SchemaColumns.GPS_LONGITUDE] = df[SchemaColumns.GPS_LONGITUDE].clip(-180, 180)

    df[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE] = df[
        SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE
    ].clip(-90, 90)
    df[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE] = df[
        SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE
    ].clip(-180, 180)

    # Validate and coerce only the schema columns
    schema_df = df[required_columns].copy()
    # THAD: FIXME.
    validated_schema_df = schema_df.copy()  # DataSchema.validate(schema_df, lazy=True)

    # Combine validated schema columns with any extra columns from original data
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns:
        result_df = pd.concat([validated_schema_df, df[extra_columns]], axis=1)
    else:
        result_df = validated_schema_df

    logger.info(
        f"Loaded {len(result_df)} records from {sqlite_db_path=} table '{table_name}'"
    )

    return result_df
