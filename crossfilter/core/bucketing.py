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
from typing import Optional

import h3
import pandas as pd

from crossfilter.core.schema import (
    SchemaColumns,
    TemporalLevel,
    get_clip_umap_h3_column_name,
    get_h3_column_name,
    get_temporal_column_name,
)

logger = logging.getLogger(__name__)


# It's important that
GROUPBY_NA_FILL_VALUE = "Unknown"


def groupby_fillna(series: pd.Series) -> pd.Series:
    """Fill NaN values in the groupby column with a placeholder value."""
    return series.fillna(GROUPBY_NA_FILL_VALUE).astype(str)


# H3 resolution levels to pre-compute (0-15, where higher = more granular).  Important that these are ordered from coarsest to finest resolution.
H3_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Temporal quantization levels.  Important that these are ordered from finest to coarsest resolution.
TEMPORAL_LEVELS = [
    TemporalLevel.SECOND,
    TemporalLevel.MINUTE,
    TemporalLevel.HOUR,
    TemporalLevel.DAY,
    TemporalLevel.MONTH,
    TemporalLevel.YEAR,
]


def add_geo_h3_bucket_columns(df: pd.DataFrame) -> None:
    """Add H3 hexagon columns at multiple resolutions."""
    logger.info(f"Adding H3 columns to DataFrame with {len(df)=} rows")
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


def add_clip_umap_h3_bucket_columns(df: pd.DataFrame) -> None:
    """Add H3 hexagon columns at multiple resolutions for CLIP UMAP coordinates."""
    logger.info(f"Adding CLIP UMAP H3 columns to DataFrame with {len(df)=} rows")
    for level in H3_LEVELS:
        col_name = get_clip_umap_h3_column_name(level)
        df[col_name] = df.apply(
            lambda row, level=level: (
                h3.latlng_to_cell(
                    row[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE],
                    row[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE],
                    level,
                )
                if pd.notna(row[SchemaColumns.CLIP_UMAP_HAVERSINE_LATITUDE])
                and pd.notna(row[SchemaColumns.CLIP_UMAP_HAVERSINE_LONGITUDE])
                else None
            ),
            axis=1,
        )


def add_temporal_bucket_columns(df: pd.DataFrame) -> None:
    """Add temporal quantization columns at multiple levels."""
    logger.info(f"Adding temporal columns to DataFrame with {len(df)=} rows.")
    timestamps: pd.Series = df[SchemaColumns.TIMESTAMP_UTC]

    # Handle empty DataFrame case early - create all empty columns and return
    if len(df) == 0:
        for level in TEMPORAL_LEVELS:
            df[get_temporal_column_name(level)] = pd.Series(
                [], dtype="datetime64[ns, UTC]"
            )
        return

    # For non-empty DataFrames, enforce that timestamps are timezone-aware and in UTC
    assert (
        timestamps.dt.tz is not None
    ), f"TIMESTAMP_UTC column must be timezone-aware, got {timestamps.dt.tz}"
    assert (
        str(timestamps.dt.tz) == "UTC"
    ), f"TIMESTAMP_UTC column must be in UTC timezone, got {timestamps.dt.tz}"

    # Basic temporal levels (second, minute, hour, day)
    df[get_temporal_column_name(TemporalLevel.SECOND)] = timestamps.dt.floor("s")
    df[get_temporal_column_name(TemporalLevel.MINUTE)] = timestamps.dt.floor("min")
    df[get_temporal_column_name(TemporalLevel.HOUR)] = timestamps.dt.floor("h")
    df[get_temporal_column_name(TemporalLevel.DAY)] = timestamps.dt.floor("D")

    # Month and year levels require special handling for period conversion
    df[get_temporal_column_name(TemporalLevel.MONTH)] = (
        timestamps.dt.tz_convert(None)
        .dt.to_period("M")
        .dt.start_time.dt.tz_localize("UTC")
    )
    df[get_temporal_column_name(TemporalLevel.YEAR)] = (
        timestamps.dt.tz_convert(None)
        .dt.to_period("Y")
        .dt.start_time.dt.tz_localize("UTC")
    )


def get_optimal_h3_level(df: pd.DataFrame, max_rows: int) -> Optional[int]:
    """
    Returns:
        Optimal H3 aggregation level, or None if no H3 aggregation is required.
    """
    if len(df.index) <= max_rows:
        logger.info(f"No need to bucket data at any H3 level, {len(df)=}, {max_rows=}")
        return None

    # Check if H3 columns are available (they won't be if GPS columns are missing)
    first_h3_column = get_h3_column_name(H3_LEVELS[0])
    if first_h3_column not in df.columns:
        logger.warning(
            "No H3 columns found in DataFrame, cannot perform H3 aggregation"
        )
        return None

    # Iterate from finest to coarsest resolution to find the finest level
    # that still produces fewer than max_rows unique buckets
    for level in reversed(H3_LEVELS):
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


def get_optimal_clip_umap_h3_level(df: pd.DataFrame, max_rows: int) -> Optional[int]:
    """
    Returns:
        Optimal CLIP UMAP H3 aggregation level, or None if no aggregation is required.
    """
    if len(df.index) <= max_rows:
        logger.info(
            f"No need to bucket data at any CLIP UMAP H3 level, {len(df)=}, {max_rows=}"
        )
        return None

    # Check if CLIP UMAP H3 columns are available
    first_clip_h3_column = get_clip_umap_h3_column_name(H3_LEVELS[0])
    if first_clip_h3_column not in df.columns:
        logger.warning(
            "No CLIP UMAP H3 columns found in DataFrame, cannot perform CLIP UMAP H3 aggregation"
        )
        return None

    # Iterate from finest to coarsest resolution to find the finest level
    # that still produces fewer than max_rows unique buckets
    for level in reversed(H3_LEVELS):
        col_name = get_clip_umap_h3_column_name(level)
        assert col_name in df.columns, f"Column {col_name} not found in DataFrame"
        unique_count = df[col_name].nunique()
        if unique_count <= max_rows:
            logger.info(
                f"Using CLIP UMAP H3 level {level} to bucket {len(df)=} rows into {max_rows=}, {unique_count=}"
            )
            return level

    logger.warning(
        f"No CLIP UMAP H3 level found that would bucket {len(df)=} rows into {max_rows=}, using {max(H3_LEVELS)}"
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


def add_temporal_bucketed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add only temporal quantization columns to the DataFrame for bucketing operations.

    This function adds temporal quantization columns based on the presence of timestamp columns.
    It is separate from H3 bucketing to allow temporal bucketing to be done at runtime
    while H3 bucketing can be done at ingestion time.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added temporal quantized columns
    """
    df = df.copy()

    # Add temporal quantization if timestamp is present
    if SchemaColumns.TIMESTAMP_UTC in df.columns:
        add_temporal_bucket_columns(df)

    logger.info(f"Added temporal bucketed columns to DataFrame with {len(df)} rows")
    return df


def add_bucketed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all quantized columns to the DataFrame for bucketing operations.

    This function combines both spatial (H3) and temporal quantization columns,
    adding them conditionally based on the presence of required input columns.

    DEPRECATED: This function is being phased out in favor of separate H3 and temporal
    bucketing functions. H3 bucketing should be done at ingestion time, while temporal
    bucketing should be done at runtime.

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
        add_geo_h3_bucket_columns(df)

    # Add temporal quantization if timestamp is present
    if SchemaColumns.TIMESTAMP_UTC in df.columns:
        add_temporal_bucket_columns(df)

    logger.info(f"Added bucketed columns to DataFrame with {len(df)} rows")
    return df


def bucket_by_target_column(
    original_data: pd.DataFrame, target_column: str, groupby_column: Optional[str]
) -> pd.DataFrame:
    """
    Create a bucketed DataFrame by grouping on the target column.

    Each row in the result represents one bucket (unique value in target_column).
    All original columns are preserved with values from the first row in each bucket.
    A COUNT column is added showing the number of original rows in each bucket.

    If we're going to later do a groupby, we need to preserve the groupby_column values independently for accurate counting.

    Returns:
        Bucketed DataFrame with one row per unique target_column value (and per groupby_column value if provided)
    """
    logger.info(
        f"bucket_by_target_column called with {target_column=}, {groupby_column=}, {id(original_data)=}"
    )
    logger.debug(f"bucket_by_target_column called with target_column='{target_column}'")
    logger.debug(f"Original data columns: {list(original_data.columns)}")
    logger.debug(
        f"Target column '{target_column}' in original data: {target_column in original_data.columns}"
    )

    if target_column not in original_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    df = original_data.copy()

    # We can't bucket on NaN values.
    df = df.dropna(subset=[target_column])

    groupby_columns = [target_column]
    if groupby_column:
        assert (
            groupby_column in original_data.columns
        ), f"Groupby column '{groupby_column}' not found in DataFrame"
        df[groupby_column] = df[groupby_column].fillna("Unknown").astype(str)
        groupby_columns.append(groupby_column)

    df[SchemaColumns.COUNT] = df.groupby(groupby_columns).transform("size")
    df = df.drop_duplicates(subset=groupby_columns)
    df = df.reset_index(drop=True)  # Keep the first.
    df.index.name = SchemaColumns.DF_ID

    logger.info(f"Setting index on {groupby_columns=}, {id(df)=}")
    df.set_index(groupby_columns, verify_integrity=True)

    return df


def filter_df_to_selected_buckets(
    original_data: pd.DataFrame,
    bucketed_df: pd.DataFrame,
    target_column: str,
    bucket_indices_to_keep: list[int],
    groupby_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter original data to only contain rows from selected buckets.

    This function implements the filtering workflow where the frontend sends
    integer row indices of selected buckets, and we filter the original data
    to only include rows that belong to those selected buckets.

    This is how we translate backwards from the df_ids in the bucketed data to the rows/df_ids in the original data.

    Returns:
        Filtered DataFrame containing only rows from selected buckets
    """
    logger.info(
        f"filter_df_to_selected_buckets called with {target_column=}, {groupby_column=}, {len(original_data)=}, {len(bucketed_df)=}, {id(original_data)=}, {id(bucketed_df)=}"
    )
    assert SchemaColumns.COUNT not in original_data.columns

    assert not pd.Series(bucket_indices_to_keep).duplicated().any()

    df = original_data.copy()
    df["original_index"] = df.index
    if target_column not in original_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in original DataFrame"
        )
    if SchemaColumns.COUNT not in bucketed_df.columns:
        logger.info(f"Adding {SchemaColumns.COUNT} column to bucketed DataFrame")
        bucketed_df[SchemaColumns.COUNT] = 1

    merge_on_columns = [target_column]

    if groupby_column:
        merge_on_columns.append(groupby_column)
        df[groupby_column] = groupby_fillna(df[groupby_column])

    assert bucketed_df[merge_on_columns].notna().all().all()
    assert bucketed_df[merge_on_columns].notna().all().all()

    logger.info(
        f"Checking index on {merge_on_columns=}, {id(bucketed_df)=}, {bucketed_df.columns=}"
    )
    # import pdb

    # pdb.set_trace()
    bucketed_df.set_index(merge_on_columns, verify_integrity=True)

    buckets_we_picked = bucketed_df.loc[
        bucket_indices_to_keep, merge_on_columns + [SchemaColumns.COUNT]
    ].set_index(merge_on_columns, verify_integrity=True)
    picked_count = buckets_we_picked[SchemaColumns.COUNT].sum()

    merged = pd.merge(
        df[merge_on_columns + ["original_index"]],
        buckets_we_picked,
        left_on=merge_on_columns,
        right_index=True,
        how="inner",
    )
    assert not merged["original_index"].duplicated().any()

    result = original_data.loc[merged["original_index"].values, :]
    assert (
        len(result) == picked_count
    ), f"{len(bucket_indices_to_keep)=} {picked_count=} , got {len(result)=}, {merge_on_columns=}"
    return result
