"""Temporal projection state management for crossfilter visualization."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    FilterOperatorType,
)
from crossfilter.core.bucketing import (
    bucket_by_target_column,
    get_optimal_temporal_level,
    get_temporal_column_name,
)
from crossfilter.core.projection_state import ProjectionState
from crossfilter.core.schema import TemporalLevel
from crossfilter.core.schema import SchemaColumns as C

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
        filtered_rows = filtered_rows.dropna(subset=[C.TIMESTAMP_UTC])

        optimal_level = get_optimal_temporal_level(
            filtered_rows, self.projection_state.max_rows
        )

        if optimal_level is None:
            self.projection_state.current_bucketing_column = None
            self.current_aggregation_level = None
            self.projection_state.projection_df = filtered_rows
            return

        self.current_aggregation_level = optimal_level
        self.projection_state.current_bucketing_column = get_temporal_column_name(
            optimal_level
        )
        self.projection_state.projection_df = bucket_by_target_column(
            filtered_rows, self.projection_state.current_bucketing_column
        )
        logger.info(
            f"Bucketed data at optimal temporal level {optimal_level=}, {len(self.projection_state.projection_df)=}"
        )

    def get_summary(self) -> dict:
        """Get a summary of the current temporal projection state."""
        summary = self.projection_state.get_summary()
        # Add temporal-specific information
        summary["aggregation_level"] = self.current_aggregation_level
        summary["target_column"] = self.projection_state.current_bucketing_column
        return summary
