"""Data quantization utilities for spatial and temporal aggregation."""

import h3
import pandas as pd
import logging
from typing import Optional, List

from crossfilter.core.schema_constants import (
    SchemaColumns,
    QuantizedColumns,
    TemporalLevel,
    DF_ID_COLUMN,
    get_h3_column_name,
)

logger = logging.getLogger(__name__)

# H3 resolution levels to pre-compute (0-15, where higher = more granular)
H3_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Temporal quantization levels
TEMPORAL_LEVELS = [
    TemporalLevel.SECOND,
    TemporalLevel.MINUTE,
    TemporalLevel.HOUR,
    TemporalLevel.DAY,
    TemporalLevel.MONTH,
    TemporalLevel.YEAR,
]


def add_quantized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all quantized columns to the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added quantized columns
    """
    df = df.copy()

    # Add spatial quantization (H3 cells)
    if (
        SchemaColumns.GPS_LATITUDE in df.columns
        and SchemaColumns.GPS_LONGITUDE in df.columns
    ):
        df = _add_h3_columns(df)

    # Add temporal quantization
    if SchemaColumns.TIMESTAMP_UTC in df.columns:
        df = _add_temporal_columns(df)

    logger.info(f"Added quantized columns to DataFrame with {len(df)} rows")
    return df


def _add_h3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add H3 hexagon columns at multiple resolutions."""
    for level in H3_LEVELS:
        col_name = get_h3_column_name(level)
        df[col_name] = df.apply(
            lambda row: (
                h3.latlng_to_cell(
                    row[SchemaColumns.GPS_LATITUDE],
                    row[SchemaColumns.GPS_LONGITUDE],
                    level,
                )
                if pd.notna(row[SchemaColumns.GPS_LATITUDE])
                and pd.notna(row[SchemaColumns.GPS_LONGITUDE])
                else None
            ),
            axis=1,
        )
    return df


def _add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal quantization columns at multiple levels."""
    # Convert to datetime if string
    timestamps = pd.to_datetime(df[SchemaColumns.TIMESTAMP_UTC])

    # Second level (round to nearest second)
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_SECOND] = timestamps.dt.floor("s")

    # Minute level
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_MINUTE] = timestamps.dt.floor("min")

    # Hour level
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_HOUR] = timestamps.dt.floor("h")

    # Day level
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_DAY] = timestamps.dt.floor("D")

    # Month level - normalize to first day of month
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_MONTH] = (
        timestamps.dt.normalize().dt.to_period("M").dt.start_time
    )

    # Year level - normalize to first day of year
    df[QuantizedColumns.QUANTIZED_TIMESTAMP_YEAR] = (
        timestamps.dt.normalize().dt.to_period("Y").dt.start_time
    )

    return df


def get_optimal_h3_level(df: pd.DataFrame, max_groups: int) -> Optional[int]:
    """
    Find the H3 resolution level that gives closest to max_groups unique cells.

    Args:
        df: DataFrame with H3 columns
        max_groups: Maximum number of groups desired

    Returns:
        Optimal H3 level, or None if no H3 columns found
    """
    best_level = None
    best_count = 0

    for level in H3_LEVELS:
        col_name = get_h3_column_name(level)
        if col_name in df.columns:
            unique_count = df[col_name].nunique()
            if unique_count <= max_groups and unique_count > best_count:
                best_level = level
                best_count = unique_count

    logger.debug(f"Optimal H3 level: {best_level} with {best_count} groups")
    return best_level


def get_optimal_temporal_level(
    df: pd.DataFrame, max_groups: int
) -> Optional[TemporalLevel]:
    """
    Find the temporal quantization level that gives closest to max_groups unique buckets.

    Args:
        df: DataFrame with temporal quantization columns
        max_groups: Maximum number of groups desired

    Returns:
        Optimal temporal level, or None if no temporal columns found
    """
    best_level = None
    best_count = 0

    for level in TEMPORAL_LEVELS:
        col_name = f"QUANTIZED_TIMESTAMP_{level.upper()}"
        if col_name in df.columns:
            unique_count = df[col_name].nunique()
            if unique_count <= max_groups and unique_count > best_count:
                best_level = level
                best_count = unique_count

    logger.debug(f"Optimal temporal level: {best_level} with {best_count} groups")
    return best_level


def aggregate_by_h3(df: pd.DataFrame, h3_level: int) -> pd.DataFrame:
    """
    Aggregate data by H3 cells at the specified level.

    Args:
        df: DataFrame with H3 columns
        h3_level: H3 resolution level to aggregate by

    Returns:
        Aggregated DataFrame with H3 cell, count, and representative lat/lon
    """
    col_name = get_h3_column_name(h3_level)
    if col_name not in df.columns:
        raise ValueError(f"H3 level {h3_level} not found in DataFrame")

    # Group by H3 cell and aggregate
    grouped = (
        df.groupby(col_name)
        .agg(
            {
                SchemaColumns.GPS_LATITUDE: ["mean", "count"],
                SchemaColumns.GPS_LONGITUDE: "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    grouped.columns = [col_name, "lat", "count", "lon"]

    # Add df_ids for the aggregated groups
    df_id_groups = (
        df.groupby(col_name)
        .apply(lambda x: list(x.index), include_groups=False)
        .reset_index(name="df_ids")
    )
    grouped = grouped.merge(df_id_groups, on=col_name)

    logger.debug(
        f"Aggregated {len(df)} rows into {len(grouped)} H3 groups at level {h3_level}"
    )
    return grouped


def aggregate_by_temporal(
    df: pd.DataFrame, temporal_level: TemporalLevel
) -> pd.DataFrame:
    """
    Aggregate data by temporal buckets at the specified level.

    Args:
        df: DataFrame with temporal quantization columns
        temporal_level: Temporal level to aggregate by

    Returns:
        Aggregated DataFrame with timestamp bucket, count, and cumulative count
    """
    col_name = f"QUANTIZED_TIMESTAMP_{temporal_level.upper()}"
    if col_name not in df.columns:
        raise ValueError(f"Temporal level {temporal_level} not found in DataFrame")

    # Group by temporal bucket and aggregate
    grouped = df.groupby(col_name).size().reset_index(name="count")

    # Add df_ids for the aggregated groups
    df_id_groups = (
        df.groupby(col_name)
        .apply(lambda x: list(x.index), include_groups=False)
        .reset_index(name="df_ids")
    )
    grouped = grouped.merge(df_id_groups, on=col_name)

    # Sort by timestamp and add cumulative count for CDF
    grouped = grouped.sort_values(col_name)
    grouped["cumulative_count"] = grouped["count"].cumsum()

    logger.debug(
        f"Aggregated {len(df)} rows into {len(grouped)} temporal groups at level {temporal_level}"
    )
    return grouped
