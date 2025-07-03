"""Base projection state management for crossfilter visualizations."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.bucketing import filter_df_to_selected_buckets
from crossfilter.core.schema import SchemaColumns

logger = logging.getLogger(__name__)


class ProjectionState:
    """
    Manages common projection state for visualizing data.
    
    This class handles the common aspects of data projections including
    the projection DataFrame, max_rows threshold, and current bucketing column.
    It provides a unified interface for filter event processing that works
    for both individual points and aggregated buckets.
    """

    def __init__(self, max_rows: int) -> None:
        """
        Initialize projection state.
        
        Args:
            max_rows: Maximum number of rows to display before aggregation
        """
        self.max_rows = max_rows
        self._projection_df = pd.DataFrame()
        self.current_bucketing_column: Optional[str] = None

    @property
    def projection_df(self) -> pd.DataFrame:
        """Get the current projection DataFrame."""
        return self._projection_df.copy()
    
    @projection_df.setter
    def projection_df(self, value: pd.DataFrame) -> None:
        """Set the current projection DataFrame."""
        self._projection_df = value

    def apply_filter_event(self, selected_df_ids: set[int], filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a filter event and return the new filtered rows.
        
        This method handles both individual points mode and aggregated buckets mode:
        - Individual points: Filter by df_id directly from the projection DataFrame
        - Aggregated buckets: Use filter_df_to_selected_buckets to map selections back to original rows
        
        Args:
            selected_df_ids: Set of df_ids selected in the visualization
            filtered_rows: Current filtered rows to apply filter to
            
        Returns:
            New filtered DataFrame containing only rows matching the selection
        """
        if not selected_df_ids:
            return pd.DataFrame()
        
        if self.current_bucketing_column is None:
            # Individual points mode - filter by df_id directly
            return filtered_rows.loc[filtered_rows.index.isin(selected_df_ids)].copy()
        
        # Aggregated mode - need to map bucket selections back to original rows
        if self.current_bucketing_column not in filtered_rows.columns:
            logger.warning(f"Bucketing column {self.current_bucketing_column} not found in filtered data")
            return filtered_rows
        
        # Get the bucket values for the selected df_ids
        selected_bucket_df_ids = list(selected_df_ids)
        
        # Make sure all selected ids are valid
        valid_ids = [id for id in selected_bucket_df_ids if id < len(self._projection_df)]
        if len(valid_ids) != len(selected_bucket_df_ids):
            logger.warning(f"Some selected df_ids are invalid: {set(selected_bucket_df_ids) - set(valid_ids)}")
        
        if not valid_ids:
            return pd.DataFrame()
        
        # Use the bucketing utility to filter original data
        try:
            return filter_df_to_selected_buckets(
                filtered_rows, 
                self._projection_df, 
                self.current_bucketing_column, 
                valid_ids
            )
        except Exception as e:
            logger.error(f"Error filtering buckets: {e}")
            return pd.DataFrame()

    def get_summary(self) -> dict:
        """Get a summary of the current projection state."""
        return {
            "max_rows": self.max_rows,
            "projection_rows": len(self._projection_df),
            "current_bucketing_column": self.current_bucketing_column,
            "is_aggregated": self.current_bucketing_column is not None,
        }