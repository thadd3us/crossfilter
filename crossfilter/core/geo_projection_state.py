"""Geographic projection state management for crossfilter visualization."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.bucketing import (
    bucket_by_target_column,
    get_h3_column_name,
    get_optimal_h3_level,
)
from crossfilter.core.projection_state import ProjectionState

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
        self.projection_state = ProjectionState(max_rows)
        self.current_h3_level: Optional[int] = None


    def update_projection(self, filtered_rows: pd.DataFrame) -> None:
        """
        Update the geographic projection based on the current filtered data.

        Args:
            filtered_rows: Current filtered subset of all_rows
        """
        optimal_level = get_optimal_h3_level(
            filtered_rows, self.projection_state.max_rows
        )

        if optimal_level is None:
            self.projection_state.current_bucketing_column = None
            self.current_h3_level = None
            self.projection_state.projection_df = filtered_rows
            return

        self.current_h3_level = optimal_level
        self.projection_state.current_bucketing_column = get_h3_column_name(
            optimal_level
        )
        self.projection_state.projection_df = bucket_by_target_column(
            filtered_rows, self.projection_state.current_bucketing_column
        )
        logger.info(
            f"Bucketed data at optimal H3 level {optimal_level=}, {len(self.projection_state.projection_df)=}"
        )


    def apply_filter_event(
        self, selected_df_ids: set[int], filtered_rows: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply a geographic filter event and return the new filtered rows.

        Args:
            selected_df_ids: Set of df_ids selected in the geographic visualization
            filtered_rows: Current filtered rows to apply filter to

        Returns:
            New filtered DataFrame containing only rows matching the selection
        """
        return self.projection_state.apply_filter_event(selected_df_ids, filtered_rows)

    def get_summary(self) -> dict:
        """Get a summary of the current geographic projection state."""
        summary = self.projection_state.get_summary()
        # Add geographic-specific information
        summary["h3_level"] = self.current_h3_level
        summary["target_column"] = self.projection_state.current_bucketing_column
        return summary
