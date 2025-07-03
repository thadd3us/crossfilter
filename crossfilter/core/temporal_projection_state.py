"""Temporal projection state management for crossfilter visualization."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.bucketing import (
    bucket_by_target_column,
    get_optimal_temporal_level,
    get_temporal_column_name,
)
from crossfilter.core.projection_state import ProjectionState
from crossfilter.core.schema import SchemaColumns, TemporalLevel

logger = logging.getLogger(__name__)


class TemporalProjectionState:
    """
    Manages the temporal projection state for visualizing data over time.
    
    This class handles the temporal aggregation of data points, maintaining
    both the current projection DataFrame and the aggregation granularity level.
    The projection may be aggregated at some granularity (with a COUNT column)
    or may show individual points (no COUNT column).
    """

    def __init__(self, max_rows: int) -> None:
        """
        Initialize temporal projection state.
        
        Args:
            max_rows: Maximum number of rows to display before aggregation
        """
        self.projection_state = ProjectionState(max_rows)
        self.current_aggregation_level: Optional[TemporalLevel] = None

    def update_projection(self, filtered_rows: pd.DataFrame) -> None:
        """
        Update the temporal projection based on the current filtered data.
        
        Args:
            filtered_rows: Current filtered subset of all_rows
        """
        if len(filtered_rows) == 0:
            self.projection_state.projection_df = pd.DataFrame()
            self.current_aggregation_level = None
            self.projection_state.current_bucketing_column = None
            logger.debug("Updated temporal projection with empty data")
            return

        if SchemaColumns.TIMESTAMP_UTC not in filtered_rows.columns:
            logger.warning("No TIMESTAMP_UTC column found in filtered data")
            self.projection_state.projection_df = pd.DataFrame()
            self.current_aggregation_level = None
            self.projection_state.current_bucketing_column = None
            return

        # If we have fewer rows than max_rows, show individual points
        if len(filtered_rows) <= self.projection_state.max_rows:
            self.projection_state.projection_df = self._create_individual_points_projection(filtered_rows)
            self.current_aggregation_level = None
            self.projection_state.current_bucketing_column = None
            logger.debug(f"Updated temporal projection with {len(self.projection_state.projection_df)} individual points")
        else:
            # Need to aggregate
            self.projection_state.projection_df = self._create_aggregated_projection(filtered_rows)
            logger.debug(f"Updated temporal projection with {len(self.projection_state.projection_df)} aggregated buckets at level {self.current_aggregation_level}")

    def _create_individual_points_projection(self, filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """Create a projection showing individual points."""
        # Return individual points sorted by timestamp
        columns_to_include = [SchemaColumns.TIMESTAMP_UTC, SchemaColumns.DATA_TYPE]

        # Include additional columns that might be useful for visualization
        for col in filtered_rows.columns:
            if col not in columns_to_include and col != SchemaColumns.DF_ID:
                columns_to_include.append(col)

        # Filter to only include columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_include if col in filtered_rows.columns]

        df = filtered_rows[existing_columns].copy()
        df = df.sort_values(SchemaColumns.TIMESTAMP_UTC)

        # Add cumulative count for CDF visualization
        df["cumulative_count"] = range(1, len(df) + 1)

        # Preserve the original df_id as a column for filtering
        df[SchemaColumns.DF_ID] = df.index

        return df

    def _create_aggregated_projection(self, filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """Create a projection with aggregated temporal buckets."""
        # Find optimal temporal level for aggregation
        optimal_level = get_optimal_temporal_level(filtered_rows, self.projection_state.max_rows)

        if optimal_level is None:
            # Fallback to least granular level
            optimal_level = TemporalLevel.YEAR

        # Get the target column name for this level
        target_column = get_temporal_column_name(optimal_level)

        if target_column not in filtered_rows.columns:
            logger.warning(f"Target column {target_column} not found in filtered data")
            return self._create_individual_points_projection(filtered_rows)

        # Create aggregated DataFrame
        bucketed = bucket_by_target_column(filtered_rows, target_column)

        # Store the aggregation details
        self.current_aggregation_level = optimal_level
        self.projection_state.current_bucketing_column = target_column

        # Transform to match expected output format for CDF visualization
        result = bucketed.rename(columns={SchemaColumns.COUNT: "count"})

        # Sort by timestamp and add cumulative count for CDF
        result = result.sort_values(target_column)
        result["cumulative_count"] = result["count"].cumsum()

        return result

    def apply_filter_event(self, selected_df_ids: set[int], filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a temporal filter event and return the new filtered rows.
        
        Args:
            selected_df_ids: Set of df_ids selected in the temporal visualization
            filtered_rows: Current filtered rows to apply filter to
            
        Returns:
            New filtered DataFrame containing only rows matching the selection
        """
        return self.projection_state.apply_filter_event(selected_df_ids, filtered_rows)

    def get_summary(self) -> dict:
        """Get a summary of the current temporal projection state."""
        summary = self.projection_state.get_summary()
        # Add temporal-specific information
        summary["aggregation_level"] = self.current_aggregation_level
        summary["target_column"] = self.projection_state.current_bucketing_column
        return summary
