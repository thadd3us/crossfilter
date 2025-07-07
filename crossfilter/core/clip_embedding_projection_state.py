"""CLIP embedding projection state management for crossfilter visualization."""

import logging
from typing import Optional

import pandas as pd

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    FilterOperatorType,
)
from crossfilter.core.bucketing import (
    bucket_by_target_column,
    get_clip_umap_h3_column_name,
    get_optimal_clip_umap_h3_level,
)
from crossfilter.core.projection_state import ProjectionState
from crossfilter.core.schema import SchemaColumns as C

logger = logging.getLogger(__name__)


class ClipEmbeddingProjectionState:
    """
    Manages the CLIP embedding projection state for visualizing semantic similarity in 2D space.

    This class handles the spatial aggregation of CLIP embedding data points using H3 hexagonal
    indexing on the UMAP-projected coordinates, maintaining both the current projection DataFrame
    and the aggregation granularity level. The projection may be aggregated at some H3 level (with
    a COUNT column) or may show individual points (no COUNT column).
    """

    def __init__(self, max_rows: int) -> None:
        """
        Initialize CLIP embedding projection state.

        Args:
            max_rows: Maximum number of rows to display before aggregation
        """
        self.projection_state = ProjectionState(max_rows)
        self.current_h3_level: Optional[int] = None

        # User selectable.
        self.max_marker_size: int = 20

    def update_projection(self, filtered_rows: pd.DataFrame) -> None:
        """
        Update the CLIP embedding projection based on the current filtered data.

        Args:
            filtered_rows: Current filtered subset of all_rows
        """
        # Drop rows with missing CLIP UMAP coordinates
        filtered_rows = filtered_rows.dropna(subset=[C.CLIP_UMAP_HAVERSINE_LATITUDE, C.CLIP_UMAP_HAVERSINE_LONGITUDE])

        optimal_level = get_optimal_clip_umap_h3_level(
            filtered_rows, self.projection_state.max_rows
        )

        if optimal_level is None:
            self.projection_state.current_bucketing_column = None
            self.current_h3_level = None
            self.projection_state.projection_df = filtered_rows
            return

        self.current_h3_level = optimal_level
        self.projection_state.current_bucketing_column = get_clip_umap_h3_column_name(
            optimal_level
        )
        self.projection_state.projection_df = bucket_by_target_column(
            filtered_rows,
            self.projection_state.current_bucketing_column,
            self.projection_state.groupby_column,
        )
        logger.info(
            f"Bucketed CLIP embedding data at optimal H3 level {optimal_level=}, {len(self.projection_state.projection_df)=}"
        )

    def get_summary(self) -> dict:
        """Get a summary of the current CLIP embedding projection state."""
        summary = self.projection_state.get_summary()
        # Add CLIP embedding-specific information
        summary["h3_level"] = self.current_h3_level
        summary["target_column"] = self.projection_state.current_bucketing_column
        return summary