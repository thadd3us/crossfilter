"""Geographic projection state management for crossfilter visualization."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.bucketing import (
    H3_LEVELS,
    bucket_by_target_column,
    get_h3_column_name,
    get_optimal_h3_level,
)
from crossfilter.core.projection_state import ProjectionState
from crossfilter.core.schema import SchemaColumns

logger = logging.getLogger(__name__)


class GeoProjectionState:
    """
    Manages the geographic projection state for visualizing spatial data.
    
    This class handles the spatial aggregation of data points using H3 hexagonal
    indexing, maintaining both the current projection DataFrame and the aggregation
    granularity level. The projection may be aggregated at some H3 level (with a COUNT
    column) or may show individual points (no COUNT column).
    """

    def __init__(self, max_rows: int) -> None:
        """
        Initialize geographic projection state.
        
        Args:
            max_rows: Maximum number of rows to display before aggregation
        """
        self._projection_state = ProjectionState(max_rows)
        self._current_h3_level: Optional[int] = None

    @property
    def max_rows(self) -> int:
        """Get the maximum number of rows before aggregation."""
        return self._projection_state.max_rows
    
    @max_rows.setter
    def max_rows(self, value: int) -> None:
        """Set the maximum number of rows before aggregation."""
        self._projection_state.max_rows = value

    @property
    def projection_df(self) -> pd.DataFrame:
        """Get the current geographic projection DataFrame."""
        return self._projection_state.projection_df.copy()

    @property
    def current_h3_level(self) -> Optional[int]:
        """Get the current H3 aggregation level, or None if showing individual points."""
        return self._current_h3_level

    @property
    def current_target_column(self) -> Optional[str]:
        """Get the current target column used for aggregation."""
        return self._projection_state.current_bucketing_column

    def update_projection(self, filtered_rows: pd.DataFrame) -> None:
        """
        Update the geographic projection based on the current filtered data.
        
        Args:
            filtered_rows: Current filtered subset of all_rows
        """
        if len(filtered_rows) == 0:
            self._projection_state.projection_df = pd.DataFrame()
            self._current_h3_level = None
            self._projection_state.current_bucketing_column = None
            logger.debug("Updated geographic projection with empty data")
            return

        # Check if we have GPS coordinates
        if (SchemaColumns.GPS_LATITUDE not in filtered_rows.columns or 
            SchemaColumns.GPS_LONGITUDE not in filtered_rows.columns):
            logger.warning("No GPS coordinates found in filtered data")
            self._projection_state.projection_df = pd.DataFrame()
            self._current_h3_level = None
            self._projection_state.current_bucketing_column = None
            return

        # If we have fewer rows than max_rows, show individual points
        if len(filtered_rows) <= self.max_rows:
            self._projection_state.projection_df = self._create_individual_points_projection(filtered_rows)
            self._current_h3_level = None
            self._projection_state.current_bucketing_column = None
            logger.debug(f"Updated geographic projection with {len(self._projection_state.projection_df)} individual points")
        else:
            # Need to aggregate
            self._projection_state.projection_df = self._create_aggregated_projection(filtered_rows)
            logger.debug(f"Updated geographic projection with {len(self._projection_state.projection_df)} aggregated buckets at H3 level {self._current_h3_level}")

    def _create_individual_points_projection(self, filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """Create a projection showing individual points."""
        # Return individual points with GPS coordinates
        columns_to_include = [SchemaColumns.GPS_LATITUDE, SchemaColumns.GPS_LONGITUDE]
        
        # Include additional columns that might be useful for visualization
        for col in filtered_rows.columns:
            if col not in columns_to_include and col != SchemaColumns.DF_ID:
                columns_to_include.append(col)
        
        # Filter to only include columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_include if col in filtered_rows.columns]
        
        df = filtered_rows[existing_columns].copy()
        
        # Preserve the original df_id as a column for filtering
        df[SchemaColumns.DF_ID] = df.index
        
        return df

    def _create_aggregated_projection(self, filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """Create a projection with aggregated H3 spatial buckets."""
        # Find optimal H3 level for aggregation
        optimal_level = get_optimal_h3_level(filtered_rows, self.max_rows)
        
        if optimal_level is None:
            # Fallback to least granular level
            optimal_level = min(H3_LEVELS)
        
        # Get the target column name for this level
        target_column = get_h3_column_name(optimal_level)
        
        if target_column not in filtered_rows.columns:
            logger.warning(f"Target column {target_column} not found in filtered data")
            return self._create_individual_points_projection(filtered_rows)
        
        # Create aggregated DataFrame
        bucketed = bucket_by_target_column(filtered_rows, target_column)
        
        # Store the aggregation details
        self._current_h3_level = optimal_level
        self._projection_state.current_bucketing_column = target_column
        
        # Transform to match expected output format for heatmap visualization
        result = bucketed.rename(columns={SchemaColumns.COUNT: "count"})
        
        return result

    def apply_filter_event(self, selected_df_ids: set[int], filtered_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a geographic filter event and return the new filtered rows.
        
        Args:
            selected_df_ids: Set of df_ids selected in the geographic visualization
            filtered_rows: Current filtered rows to apply filter to
            
        Returns:
            New filtered DataFrame containing only rows matching the selection
        """
        return self._projection_state.apply_filter_event(selected_df_ids, filtered_rows)

    def get_summary(self) -> dict:
        """Get a summary of the current geographic projection state."""
        summary = self._projection_state.get_summary()
        # Add geographic-specific information
        summary["h3_level"] = self._current_h3_level
        summary["target_column"] = self._projection_state.current_bucketing_column
        return summary