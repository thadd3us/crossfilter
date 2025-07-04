"""
Data bucketing utilities for spatial and temporal aggregation.

This module provides functionality to:
1. Add precomputed quantized columns (H3 spatial cells, temporal buckets) to DataFrames
2. Create bucketed DataFrames by groupby operations on target columns
3. Filter original data based on selected buckets from the frontend

## Bucketing Design

The core bucketing operation takes original data and a target column (which could be
a precomputed quantized column) and produces a bucketed DataFrame where:

- Each row represents one bucket (unique value in the target column)
- All original columns are preserved, filled with values from the first row in each bucket
- A COUNT column is added showing how many original rows fell into this bucket
- DF_ID uses standard Pandas int index (not the original df_ids)

## Filtering Workflow

When the frontend reports that certain bucket rows are selected:
1. Frontend sends integer row indices of selected buckets
2. We look up the corresponding target column values from the bucketed DataFrame
3. We filter the original data using: original_data[target_column].isin(selected_values)

This approach eliminates the need to track df_ids lists per bucket, simplifying
the data structure while maintaining full filtering capability.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import h3
import pandas as pd

from crossfilter.core.schema import (
    SchemaColumns,
    TemporalLevel,
    get_h3_column_name,
    get_temporal_column_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucketKey:
    """
    Represents a specific bucketing configuration for data aggregation.

    When the frontend sends df_ids from bucketed data, this key identifies
    which bucketing operation was used to ensure we apply the correct
    filtering logic.
    """

    target_column: str
    identifier: str = ""

    def __post_init__(self) -> None:
        """Generate a unique identifier if not provided."""
        if not self.identifier:
            # Generate a short UUID for this bucketing instance
            object.__setattr__(self, "identifier", str(uuid.uuid4())[:8])


# H3 resolution levels to pre-compute (0-15, where higher = more granular).  Important that these are ordered from smallest to largest grain.
H3_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Temporal quantization levels.  Important that these are ordered from smallest to largest grain.
TEMPORAL_LEVELS = [
    TemporalLevel.SECOND,
    TemporalLevel.MINUTE,
    TemporalLevel.HOUR,
    TemporalLevel.DAY,
    TemporalLevel.MONTH,
    TemporalLevel.YEAR,
]


def add_quantized_geo_h3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add H3 spatial quantization columns to the DataFrame.

    Args:
        df: Input DataFrame with GPS_LATITUDE and GPS_LONGITUDE columns

    Returns:
        DataFrame with added H3 quantization columns
    """
    if not (
        SchemaColumns.GPS_LATITUDE in df.columns
        and SchemaColumns.GPS_LONGITUDE in df.columns
    ):
        raise ValueError(
            "DataFrame must contain GPS_LATITUDE and GPS_LONGITUDE columns"
        )

    df = df.copy()
    df = _add_h3_columns(df)
    logger.info(f"Added H3 quantization columns to DataFrame with {len(df)} rows")
    return df


def add_quantized_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal quantization columns to the DataFrame.

    Args:
        df: Input DataFrame with TIMESTAMP_UTC column

    Returns:
        DataFrame with added temporal quantization columns
    """
    if SchemaColumns.TIMESTAMP_UTC not in df.columns:
        raise ValueError("DataFrame must contain TIMESTAMP_UTC column")

    df = df.copy()
    df = _add_temporal_columns(df)
    logger.info(f"Added temporal quantization columns to DataFrame with {len(df)} rows")
    return df


def _add_h3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add H3 hexagon columns at multiple resolutions."""
    for level in H3_LEVELS:
        col_name = get_h3_column_name(level)
        df[col_name] = df.apply(
            lambda row, level=level: (
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
    df[get_temporal_column_name(TemporalLevel.SECOND)] = timestamps.dt.floor("s")

    # Minute level
    df[get_temporal_column_name(TemporalLevel.MINUTE)] = timestamps.dt.floor("min")

    # Hour level
    df[get_temporal_column_name(TemporalLevel.HOUR)] = timestamps.dt.floor("h")

    # Day level
    df[get_temporal_column_name(TemporalLevel.DAY)] = timestamps.dt.floor("D")

    # Month level - normalize to first day of month
    df[get_temporal_column_name(TemporalLevel.MONTH)] = (
        timestamps.dt.normalize().dt.to_period("M").dt.start_time
    )

    # Year level - normalize to first day of year
    df[get_temporal_column_name(TemporalLevel.YEAR)] = (
        timestamps.dt.normalize().dt.to_period("Y").dt.start_time
    )

    return df


def bucket_by_target_column(
    original_data: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """
    Create a bucketed DataFrame by grouping on the target column.

    Each row in the result represents one bucket (unique value in target_column).
    All original columns are preserved with values from the first row in each bucket.
    A COUNT column is added showing the number of original rows in each bucket.

    Args:
        original_data: The original DataFrame to bucket
        target_column: Column name to group by (should be a quantized/discretized column)

    Returns:
        Bucketed DataFrame with one row per unique target_column value
    """
    logger.debug(f"bucket_by_target_column called with target_column='{target_column}'")
    logger.debug(f"Original data columns: {list(original_data.columns)}")
    logger.debug(
        f"Target column '{target_column}' in original data: {target_column in original_data.columns}"
    )

    if target_column not in original_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Group by target column and take first value for each column, plus count
    # Exclude the target column from aggregation since it becomes the index
    agg_dict = {}
    for col in original_data.columns:
        if col != target_column:
            agg_dict[col] = "first"

    # Get the grouped data, including nulls in groupby
    logger.debug(f"Aggregation dictionary: {agg_dict}")
    grouped = original_data.groupby(target_column, dropna=False).agg(agg_dict)
    logger.debug(f"Grouped DataFrame columns: {list(grouped.columns)}")
    logger.debug(
        f"Target column '{target_column}' in grouped DataFrame (index): {target_column in grouped.index.names}"
    )

    # Add count column
    counts = original_data.groupby(target_column, dropna=False).size()

    # Combine the grouped data with counts
    bucketed = grouped.copy()
    bucketed[SchemaColumns.COUNT] = counts

    # Reset index to make target_column a regular column
    bucketed = bucketed.reset_index()
    logger.debug(f"Final bucketed DataFrame columns: {list(bucketed.columns)}")
    logger.debug(
        f"Target column '{target_column}' in final bucketed DataFrame: {target_column in bucketed.columns}"
    )

    # Set standard integer index for DF_ID
    bucketed.index.name = SchemaColumns.DF_ID

    logger.debug(
        f"Bucketed {len(original_data)} rows into {len(bucketed)} buckets by column '{target_column}'"
    )
    logger.debug(f"Bucketed DataFrame columns: {list(bucketed.columns)}")
    logger.debug(
        f"Target column '{target_column}' in bucketed DataFrame: {target_column in bucketed.columns}"
    )

    return bucketed


def get_optimal_h3_level(df: pd.DataFrame, max_rows: int) -> Optional[int]:
    """
    Returns:
        Optimal H3 aggregation level, or None if no H3 aggregation is required.
    """
    if len(df.index) <= max_rows:
        logger.info(
            f"No need to bucket data at any H3 level, {len(df)=}, {max_rows=}"
        )
        return None

    # Check if H3 columns are available (they won't be if GPS columns are missing)
    first_h3_column = get_h3_column_name(H3_LEVELS[0])
    if first_h3_column not in df.columns:
        logger.warning(
            f"No H3 columns found in DataFrame, cannot perform H3 aggregation"
        )
        return None

    for level in H3_LEVELS:
        col_name = get_h3_column_name(level)
        assert col_name in df.columns, f"Column {col_name} not found in DataFrame"
        unique_count = df[col_name].nunique()
        if unique_count <= max_rows:
            logger.info(
                f"Using H3 level {level} to bucket {len(df)=} rows into {max_rows=}, {unique_count=}"
            )
            return level

    logger.warning(
        f"No H3 level found that would bucket {len(df)=} rows into {max_rows=}, using {max(H3_LEVELS)}"
    )
    return max(H3_LEVELS)


def get_optimal_temporal_level(
    df: pd.DataFrame, max_rows: int
) -> Optional[TemporalLevel]:
    """
    Returns:
        Optimal temporal aggregation level, or None if no temporal aggregation is required.
    """
    if len(df.index) <= max_rows:
        logger.info(
            f"No need to bucket data at any temporal level, {len(df)=}, {max_rows=}"
        )
        return None

    for level in TEMPORAL_LEVELS:
        col_name = get_temporal_column_name(level)
        assert col_name in df.columns, f"Column {col_name} not found in DataFrame"
        unique_count = df[col_name].nunique()
        if unique_count <= max_rows:
            logger.info(
                f"Using temporal level {level} to bucket {len(df)=} rows into {max_rows=}, {unique_count=}"
            )
            return level

    logger.warning(
        f"No temporal level found that would bucket {len(df)=} rows into {max_rows=}, using {TemporalLevel.YEAR}"
    )
    return TemporalLevel.YEAR


def filter_df_to_selected_buckets(
    original_data: pd.DataFrame,
    bucketed_df: pd.DataFrame,
    target_column: str,
    bucket_indices_to_keep: list[int],
) -> pd.DataFrame:
    """
    Filter original data to only contain rows from selected buckets.

    This function implements the filtering workflow where the frontend sends
    integer row indices of selected buckets, and we filter the original data
    to only include rows that belong to those selected buckets.

    Args:
        original_data: The original, unbucketed DataFrame
        bucketed_df: The bucketed DataFrame (output from bucket_by_target_column)
        target_column: Column name that was used for bucketing (present in both DataFrames)
        bucket_indices_to_keep: List of integer indices from bucketed_df to keep

    Returns:
        Filtered DataFrame containing only rows from selected buckets
    """
    if target_column not in original_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in original DataFrame"
        )

    if target_column not in bucketed_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in bucketed DataFrame"
        )

    # Validate bucket indices
    invalid_indices = [
        idx for idx in bucket_indices_to_keep if idx < 0 or idx >= len(bucketed_df)
    ]
    if invalid_indices:
        raise ValueError(
            f"Invalid bucket indices: {invalid_indices}. Valid range is 0-{len(bucketed_df)-1}"
        )

    # Get the target column values for selected buckets as Series
    selected_bucket_values = bucketed_df.iloc[bucket_indices_to_keep][target_column]

    # Check that none of the selected bucket values is NaN
    if selected_bucket_values.isna().any():
        raise ValueError(
            "Selected bucket values cannot contain NaN. This indicates invalid bucketing."
        )

    # Filter original data using isin
    mask = original_data[target_column].isin(selected_bucket_values)
    filtered_data = original_data[mask].copy()

    return filtered_data


def add_bucketed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all quantized columns to the DataFrame for bucketing operations.

    This function combines both spatial (H3) and temporal quantization columns,
    adding them conditionally based on the presence of required input columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added quantized columns for both spatial and temporal data
    """
    df = df.copy()

    # Add spatial quantization (H3 cells) if GPS coordinates are present
    if (
        SchemaColumns.GPS_LATITUDE in df.columns
        and SchemaColumns.GPS_LONGITUDE in df.columns
    ):
        df = _add_h3_columns(df)

    # Add temporal quantization if timestamp is present
    if SchemaColumns.TIMESTAMP_UTC in df.columns:
        df = _add_temporal_columns(df)

    logger.info(f"Added bucketed columns to DataFrame with {len(df)} rows")
    return df


def bucket_dataframe(bucket_key: BucketKey, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a bucketed DataFrame using the specified bucketing configuration.

    This function takes a BucketKey that specifies the bucketing type, target column,
    and identifier, and returns a bucketed DataFrame that can be sent to the frontend.

    Args:
        bucket_key: Configuration specifying how to bucket the data
        df: DataFrame to bucket (should have quantized columns pre-computed)

    Returns:
        Bucketed DataFrame with one row per unique bucket value

    Raises:
        ValueError: If the target column is not found in the DataFrame
    """
    return bucket_by_target_column(df, bucket_key.target_column)
