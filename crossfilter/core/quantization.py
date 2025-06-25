"""Data quantization utilities for spatial and temporal aggregation."""

import h3
import pandas as pd
from datetime import datetime
from typing import Optional


class DataQuantizer:
    """
    Handles pre-computation of quantized columns for spatial and temporal data.
    
    This class adds quantized columns to the DataFrame that enable efficient
    grouping and aggregation for visualization at different zoom levels.
    """
    
    # H3 resolution levels to pre-compute (0-15, where higher = more granular)
    H3_LEVELS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Temporal quantization levels
    TEMPORAL_LEVELS = [
        'second', 'minute', 'hour', 'day', 'month', 'year'
    ]
    
    @classmethod
    def add_quantized_columns(cls, df: pd.DataFrame, 
                            lat_col: str = 'GPS_LATITUDE',
                            lon_col: str = 'GPS_LONGITUDE', 
                            timestamp_col: str = 'TIMESTAMP_UTC') -> pd.DataFrame:
        """
        Add all quantized columns to the DataFrame.
        
        Args:
            df: Input DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column  
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with added quantized columns
        """
        df = df.copy()
        
        # Add spatial quantization (H3 cells)
        if lat_col in df.columns and lon_col in df.columns:
            df = cls._add_h3_columns(df, lat_col, lon_col)
        
        # Add temporal quantization
        if timestamp_col in df.columns:
            df = cls._add_temporal_columns(df, timestamp_col)
            
        return df
    
    @classmethod
    def _add_h3_columns(cls, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
        """Add H3 hexagon columns at multiple resolutions."""
        for level in cls.H3_LEVELS:
            col_name = f"QUANTIZED_H3_L{level}"
            df[col_name] = df.apply(
                lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], level)
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col])
                else None,
                axis=1
            )
        return df
    
    @classmethod  
    def _add_temporal_columns(cls, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add temporal quantization columns at multiple levels."""
        # Convert to datetime if string
        timestamps = pd.to_datetime(df[timestamp_col])
        
        # Second level (round to nearest second)
        df['QUANTIZED_TIMESTAMP_SECOND'] = timestamps.dt.floor('s')
        
        # Minute level  
        df['QUANTIZED_TIMESTAMP_MINUTE'] = timestamps.dt.floor('min')
        
        # Hour level
        df['QUANTIZED_TIMESTAMP_HOUR'] = timestamps.dt.floor('h')
        
        # Day level
        df['QUANTIZED_TIMESTAMP_DAY'] = timestamps.dt.floor('D')
        
        # Month level - normalize to first day of month
        df['QUANTIZED_TIMESTAMP_MONTH'] = timestamps.dt.normalize().dt.to_period('M').dt.start_time
        
        # Year level - normalize to first day of year
        df['QUANTIZED_TIMESTAMP_YEAR'] = timestamps.dt.normalize().dt.to_period('Y').dt.start_time
        
        return df
    
    @classmethod
    def get_optimal_h3_level(cls, df: pd.DataFrame, max_groups: int = 100000) -> Optional[int]:
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
        
        for level in cls.H3_LEVELS:
            col_name = f"QUANTIZED_H3_L{level}"
            if col_name in df.columns:
                unique_count = df[col_name].nunique()
                if unique_count <= max_groups and unique_count > best_count:
                    best_level = level
                    best_count = unique_count
        
        return best_level
    
    @classmethod
    def get_optimal_temporal_level(cls, df: pd.DataFrame, max_groups: int = 100000) -> Optional[str]:
        """
        Find the temporal quantization level that gives closest to max_groups unique buckets.
        
        Args:
            df: DataFrame with temporal quantization columns
            max_groups: Maximum number of groups desired
            
        Returns:
            Optimal temporal level name, or None if no temporal columns found
        """
        best_level = None
        best_count = 0
        
        for level in cls.TEMPORAL_LEVELS:
            col_name = f"QUANTIZED_TIMESTAMP_{level.upper()}"
            if col_name in df.columns:
                unique_count = df[col_name].nunique()
                if unique_count <= max_groups and unique_count > best_count:
                    best_level = level
                    best_count = unique_count
        
        return best_level
    
    @classmethod
    def aggregate_by_h3(cls, df: pd.DataFrame, h3_level: int) -> pd.DataFrame:
        """
        Aggregate data by H3 cells at the specified level.
        
        Args:
            df: DataFrame with H3 columns
            h3_level: H3 resolution level to aggregate by
            
        Returns:
            Aggregated DataFrame with H3 cell, count, and representative lat/lon
        """
        col_name = f"QUANTIZED_H3_L{h3_level}"
        if col_name not in df.columns:
            raise ValueError(f"H3 level {h3_level} not found in DataFrame")
        
        # Group by H3 cell and aggregate
        grouped = df.groupby(col_name).agg({
            'GPS_LATITUDE': ['mean', 'count'],
            'GPS_LONGITUDE': 'mean',
            'UUID_LONG': lambda x: list(x)  # Keep track of original IDs
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [col_name, 'lat', 'count', 'lon', 'uuids']
        
        return grouped
    
    @classmethod
    def aggregate_by_temporal(cls, df: pd.DataFrame, temporal_level: str) -> pd.DataFrame:
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
        grouped = df.groupby(col_name).agg({
            'UUID_LONG': ['count', lambda x: list(x)]
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [col_name, 'count', 'uuids']
        
        # Sort by timestamp and add cumulative count for CDF
        grouped = grouped.sort_values(col_name)
        grouped['cumulative_count'] = grouped['count'].cumsum()
        
        return grouped