"""Session state management for single-session Crossfilter application."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import pandas as pd

from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import FilterEvent, ProjectionType
from crossfilter.core.temporal_projection_state import TemporalProjectionState

logger = logging.getLogger(__name__)


class SessionState:
    """
    Manages the current state of data loaded into the Crossfilter application.

    This class represents the single session state for the application, holding
    the currently loaded dataset and managing projections of the data for
    crossfiltering and visualization.

    In the single-session design pattern, there is exactly one instance of this
    class per web server instance, simplifying state management and eliminating
    the need for complex multi-user session handling.

    The SessionState manages a set of data projections (temporal, geographic, etc.) that
    maintain their own visualization state and aggregation levels.
    """

    def __init__(self, default_max_rows: int = 100000) -> None:
        """Initialize session state with empty DataFrame."""
        # Complete unaggregated dataset with pre-computed quantized columns for H3 spatial cells and temporal buckets
        self.all_rows = pd.DataFrame()
        # Current filtered subset of all_rows after filtering operations
        self.filtered_rows = pd.DataFrame()
        self.update_metadata()

        # Initialize projection states
        self.temporal_projection = TemporalProjectionState(max_rows=default_max_rows)
        self.geo_projection = GeoProjectionState(max_rows=default_max_rows)

        # SSE event broadcasting
        self.filter_version = 0
        self.sse_clients: set[asyncio.Queue] = set()


    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the session state."""
        # Add bucketed columns and store as all_rows
        bucketed_data = add_bucketed_columns(df)
        self.all_rows = bucketed_data

        # Initialize filtered_rows to show all data
        self.filtered_rows = self.all_rows.copy()

        self.update_metadata()
        logger.info(
            f"Loaded dataset with {len(df)} rows (expanded to {len(bucketed_data)} rows with bucketed columns) into session state"
        )

        # Update all projections with the new data
        self.update_all_projections()

        # Broadcast data loaded event
        self.broadcast_filter_change("data_loaded", ["temporal", "geo"])

    def update_metadata(self) -> None:
        """Update metadata based on current DataFrame."""
        self.metadata = {
            "shape": self.all_rows.shape,
            "columns": list(self.all_rows.columns),
            "dtypes": self.all_rows.dtypes.to_dict(),
        }

    def has_data(self) -> bool:
        """Check if the session has data loaded."""
        return not self.all_rows.empty

    def clear(self) -> None:
        """Clear all data from the session state."""
        self.all_rows = pd.DataFrame()
        self.filtered_rows = pd.DataFrame()
        self.update_metadata()

        # Update all projections with empty data
        self.update_all_projections()

    def update_all_projections(self) -> None:
        """Update all projection states with the current filtered data."""
        self.temporal_projection.update_projection(self.filtered_rows)
        self.geo_projection.update_projection(self.filtered_rows)

    def apply_filter_event(self, filter_event: FilterEvent) -> None:
        """
        Apply a filter event from a specific projection.

        Args:
            filter_event: The filter event containing projection type and selected df_ids
        """
        if filter_event.projection_type == ProjectionType.TEMPORAL:
            new_filtered_rows = self.temporal_projection.apply_filter_event(
                filter_event.selected_df_ids, self.filtered_rows
            )
        elif filter_event.projection_type == ProjectionType.GEO:
            new_filtered_rows = self.geo_projection.apply_filter_event(
                filter_event.selected_df_ids, self.filtered_rows
            )
        else:
            logger.warning(f"Unknown projection type: {filter_event.projection_type}")
            return

        # Update filtered_rows with the new selection
        self.filtered_rows = new_filtered_rows

        # Update all projections with the new filtered data
        self.update_all_projections()

        # Broadcast filter change event
        self.broadcast_filter_change("filter_applied", ["temporal", "geo"])

    def reset_filters(self) -> None:
        """Reset all filters to show all data."""
        if len(self.filtered_rows) != len(self.all_rows):
            self.filtered_rows = self.all_rows.copy()
            self.update_all_projections()
            self.broadcast_filter_change("filter_reset", ["temporal", "geo"])
            logger.info("Reset all filters - all points now visible")

    def get_summary(self) -> dict:
        """Get a summary of the current session state."""
        if not self.has_data():
            return {
                "status": "empty",
                "message": "No data loaded",
                "all_rows_count": 0,
                "filtered_rows_count": 0,
                "columns": [],
            }

        summary = {
            "status": "loaded",
            "shape": self.metadata["shape"],
            "columns": self.metadata["columns"],
            "memory_usage": f"{self.all_rows.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "all_rows_count": len(self.all_rows),
            "filtered_rows_count": len(self.filtered_rows),
            "filter_ratio": (
                len(self.filtered_rows) / len(self.all_rows)
                if len(self.all_rows) > 0
                else 0
            ),
        }

        # Add projection state info
        summary["temporal_projection"] = self.temporal_projection.get_summary()
        summary["geo_projection"] = self.geo_projection.get_summary()

        return summary

    def get_geo_aggregation(self) -> pd.DataFrame:
        return self.geo_projection.projection_state.projection_df.copy()

    def get_temporal_projection(self) -> pd.DataFrame:
        return self.temporal_projection.projection_state.projection_df.copy()

    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get the current filtered dataset.

        Returns:
            Current filtered subset of the data
        """
        return self.filtered_rows.copy()


    def broadcast_filter_change(
        self, event_type: str, affected_components: list[str]
    ) -> None:
        """Broadcast filter change event to all SSE clients."""
        self.filter_version += 1
        event_data = {
            "type": event_type,
            "affected_components": affected_components,
            "version": self.filter_version,
            "timestamp": time.time(),
            "session_state": {
                "has_data": self.has_data(),
                "all_rows_count": len(self.all_rows) if self.has_data() else 0,
                "filtered_rows_count": (
                    len(self.filtered_rows) if self.has_data() else 0
                ),
                "columns": list(self.all_rows.columns) if self.has_data() else [],
                # Frontend-compatible field names
                "row_count": len(self.all_rows) if self.has_data() else 0,
                "filtered_count": len(self.filtered_rows) if self.has_data() else 0,
            },
        }

        # Queue the event for all connected SSE clients
        for client_queue in self.sse_clients:
            try:
                client_queue.put_nowait(event_data)
            except asyncio.QueueFull:
                logger.warning("SSE client queue full, dropping event")

    async def filter_change_stream(self) -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events stream for filter changes."""
        client_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.sse_clients.add(client_queue)

        try:
            # Send initial connection event
            initial_event = {
                "type": "connection_established",
                "affected_components": ["temporal", "geo"],
                "version": self.filter_version,
                "timestamp": time.time(),
                "session_state": {
                    "has_data": self.has_data(),
                    "all_rows_count": len(self.all_rows) if self.has_data() else 0,
                    "filtered_rows_count": (
                        len(self.filtered_rows) if self.has_data() else 0
                    ),
                    "columns": list(self.all_rows.columns) if self.has_data() else [],
                    # Frontend-compatible field names
                    "row_count": len(self.all_rows) if self.has_data() else 0,
                    "filtered_count": (
                        len(self.filtered_rows) if self.has_data() else 0
                    ),
                },
            }
            yield f"data: {json.dumps(initial_event)}\n\n"

            # Stream subsequent events
            while True:
                try:
                    # Reduced timeout from 30s to 2s for faster test shutdown
                    event_data = await asyncio.wait_for(client_queue.get(), timeout=2.0)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.TimeoutError:
                    # Send periodic heartbeat to keep connection alive
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                        "version": self.filter_version,
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE client disconnected")
        finally:
            self.sse_clients.discard(client_queue)
