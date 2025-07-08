"""Base projection state management for crossfilter visualizations."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    FilterOperatorType,
)
from crossfilter.core.bucketing import filter_df_to_selected_buckets
from crossfilter.core.schema import SchemaColumns as C

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
        self.projection_df = pd.DataFrame()
        self.current_bucketing_column: Optional[str] = None
        self.groupby_column: Optional[str] = str(C.DATA_TYPE)

    def apply_filter_event(
        self, filter_event: FilterEvent, filtered_rows: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply a filter event and return the new filtered rows.

        This method handles both individual points mode and aggregated buckets mode:
        - Individual points: Filter by df_id directly from the projection DataFrame
        - Aggregated buckets: Use filter_df_to_selected_buckets to map selections back to original rows

        Args:
            filter_event: FilterEvent containing selected df_ids and filter operation
            filtered_rows: Current filtered rows to apply filter to

        Returns:
            New filtered DataFrame containing only rows matching the selection
        """
        logger.info(
            f"Applying filter event: {filter_event.filter_operator=}, {filter_event.projection_type=}, {self.current_bucketing_column=}, {self.groupby_column=}, {len(self.projection_df)=}"
        )
        if not filter_event.selected_df_ids:
            return pd.DataFrame()

        # Get the selected rows based on the projection mode
        if self.current_bucketing_column is None:
            # Individual points mode - filter by df_id directly
            selected_rows = filtered_rows.loc[
                filtered_rows.index.isin(filter_event.selected_df_ids)
            ]
        else:
            # Aggregated mode - need to map bucket selections back to original rows
            if self.current_bucketing_column not in filtered_rows.columns:
                raise ValueError(
                    f"Current bucketing column {self.current_bucketing_column} not found in filtered rows"
                )

            # Get the bucket values for the selected df_ids
            selected_bucket_df_ids = list(filter_event.selected_df_ids)

            # Use the bucketing utility to filter original data
            selected_rows = filter_df_to_selected_buckets(
                filtered_rows,
                self.projection_df,
                self.current_bucketing_column,
                bucket_indices_to_keep=selected_bucket_df_ids,
                # BUG is fixed, but not tested.  Is this code covered?
                groupby_column=self.groupby_column,
            )

        # Apply the filter operation
        if filter_event.filter_operator == FilterOperatorType.INTERSECTION:
            # Keep only the selected rows
            return selected_rows.copy()
        elif filter_event.filter_operator == FilterOperatorType.SUBTRACTION:
            # Remove the selected rows from the filtered data
            # THAD: Make sure this line is covered by tests.
            return filtered_rows.drop(selected_rows.index).copy()
        else:
            raise ValueError(f"Invalid filter operator: {filter_event.filter_operator}")

    def get_summary(self) -> dict:
        """Get a summary of the current projection state."""
        return {
            "max_rows": self.max_rows,
            "projection_rows": len(self.projection_df),
            "current_bucketing_column": self.current_bucketing_column,
            "is_aggregated": self.current_bucketing_column is not None,
            "groupby_column": self.groupby_column,
        }
