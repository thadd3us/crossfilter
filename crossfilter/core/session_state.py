"""Session state management for single-session Crossfilter application."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import pandas as pd

from crossfilter.core.backend_frontend_shared_schema import (
    FilterEvent,
    ProjectionType,
    SessionStateResponse,
)
from crossfilter.core.schema import SchemaColumns as C
from crossfilter.core.bucketing import add_temporal_bucketed_columns
from crossfilter.core.geo_projection_state import GeoProjectionState
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
    maintain their own visualization state and bucketing levels.
    """

    def __init__(self) -> None:
        """Initialize session state with empty DataFrame."""
        # Complete unaggregated dataset with pre-computed quantized columns for H3 spatial cells and temporal buckets
        self.all_rows = pd.DataFrame()
        # Current filtered subset of all_rows after filtering operations
        self.filtered_rows = pd.DataFrame()

        # Initialize projection states
        self.temporal_projection = TemporalProjectionState(max_rows=10_000)
        self.geo_projection = GeoProjectionState(max_rows=10_000)

        # SSE event broadcasting
        self.filter_version = 0
        self.sse_clients: set[asyncio.Queue] = set()

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the session state."""
        # Add temporal bucketed columns at runtime (H3 columns should already be present from ingestion)
        bucketed_data = add_temporal_bucketed_columns(df)
        self.all_rows = bucketed_data

        # Initialize filtered_rows to show all data
        self.filtered_rows = self.all_rows.copy()

        logger.info(
            f"Loaded dataset with {len(df)} rows (expanded to {len(bucketed_data)} rows with temporal bucketed columns) into session state"
        )

        # Update all projections with the new data
        self.update_all_projections()

        # Broadcast data loaded event
        self.broadcast_filter_change("data_loaded", ["temporal", "geo"])

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
            projection_state = self.temporal_projection.projection_state
        elif filter_event.projection_type == ProjectionType.GEO:
            projection_state = self.geo_projection.projection_state
        else:
            logger.error(f"Invalid projection type: {filter_event.projection_type}")
            raise ValueError(f"Invalid projection type: {filter_event.projection_type}")

        new_filtered_rows = projection_state.apply_filter_event(
            filter_event, self.filtered_rows
        )
        assert C.TIMESTAMP_UTC in new_filtered_rows.columns

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
        return {
            "columns": list(self.all_rows.columns),
            "memory_usage": f"{self.all_rows.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "all_rows_count": len(self.all_rows),
            "filtered_rows_count": len(self.filtered_rows),
            "temporal_projection": self.temporal_projection.get_summary(),
            "geo_projection": self.geo_projection.get_summary(),
        }

    def _create_session_state_response(self) -> SessionStateResponse:
        """Create a strongly typed session state response."""
        all_rows_count = len(self.all_rows)
        filtered_rows_count = len(self.filtered_rows)
        return SessionStateResponse(
            all_rows_count=all_rows_count,
            filtered_rows_count=filtered_rows_count,
            columns=list(self.all_rows.columns),
            memory_usage_mb=f"{self.all_rows.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
            temporal_projection=self.temporal_projection.get_summary(),
            geo_projection=self.geo_projection.get_summary(),
            has_data=all_rows_count > 0,
            row_count=all_rows_count,
            filtered_count=filtered_rows_count,
        )

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
            "session_state": self._create_session_state_response().model_dump(),
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
                "session_state": self._create_session_state_response().model_dump(),
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
