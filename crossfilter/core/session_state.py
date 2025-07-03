"""Session state management for single-session Crossfilter application."""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import pandas as pd

from crossfilter.core.bucketing import add_bucketed_columns
from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import FilterEvent, ProjectionType, SchemaColumns
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
        self._all_rows = pd.DataFrame()
        # Current filtered subset of all_rows after filtering operations
        self._filtered_rows = pd.DataFrame()
        self._update_metadata()
        
        # Initialize projection states
        self._temporal_projection = TemporalProjectionState(max_rows=default_max_rows)
        self._geo_projection = GeoProjectionState(max_rows=default_max_rows)
        
        # SSE event broadcasting
        self._filter_version = 0
        self._sse_clients: set[asyncio.Queue] = set()

    @property
    def all_rows(self) -> pd.DataFrame:
        """Get the complete dataset with all rows."""
        return self._all_rows.copy()

    @property
    def filtered_rows(self) -> pd.DataFrame:
        """Get the currently filtered dataset."""
        return self._filtered_rows.copy()

    @property
    def temporal_projection(self) -> TemporalProjectionState:
        """Get the temporal projection state."""
        return self._temporal_projection

    @property
    def geo_projection(self) -> GeoProjectionState:
        """Get the geographic projection state."""
        return self._geo_projection

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the session state."""
        # Add bucketed columns and store as all_rows
        bucketed_data = add_bucketed_columns(df)
        self._all_rows = bucketed_data
        
        # Initialize filtered_rows to show all data
        self._filtered_rows = self._all_rows.copy()
        
        self._update_metadata()
        logger.info(f"Loaded dataset with {len(df)} rows (expanded to {len(bucketed_data)} rows with bucketed columns) into session state")
        
        # Update all projections with the new data
        self._update_all_projections()
        
        # Broadcast data loaded event
        self._broadcast_filter_change("data_loaded", ["temporal", "geo"])

    def _update_metadata(self) -> None:
        """Update metadata based on current DataFrame."""
        self._metadata = {
            "shape": self._all_rows.shape,
            "columns": list(self._all_rows.columns),
            "dtypes": self._all_rows.dtypes.to_dict(),
        }

    @property
    def metadata(self) -> dict:
        """Get metadata about the current dataset."""
        return self._metadata.copy()

    def has_data(self) -> bool:
        """Check if the session has data loaded."""
        return not self._all_rows.empty

    def clear(self) -> None:
        """Clear all data from the session state."""
        self._all_rows = pd.DataFrame()
        self._filtered_rows = pd.DataFrame()
        self._update_metadata()
        
        # Update all projections with empty data
        self._update_all_projections()

    def _update_all_projections(self) -> None:
        """Update all projection states with the current filtered data."""
        self._temporal_projection.update_projection(self._filtered_rows)
        self._geo_projection.update_projection(self._filtered_rows)

    def apply_filter_event(self, filter_event: FilterEvent) -> None:
        """
        Apply a filter event from a specific projection.
        
        Args:
            filter_event: The filter event containing projection type and selected df_ids
        """
        if filter_event.projection_type == ProjectionType.TEMPORAL:
            new_filtered_rows = self._temporal_projection.apply_filter_event(
                filter_event.selected_df_ids, self._filtered_rows
            )
        elif filter_event.projection_type == ProjectionType.GEO:
            new_filtered_rows = self._geo_projection.apply_filter_event(
                filter_event.selected_df_ids, self._filtered_rows
            )
        else:
            logger.warning(f"Unknown projection type: {filter_event.projection_type}")
            return
        
        # Update filtered_rows with the new selection
        self._filtered_rows = new_filtered_rows
        
        # Update all projections with the new filtered data
        self._update_all_projections()
        
        # Broadcast filter change event
        self._broadcast_filter_change("filter_applied", ["temporal", "geo"])

    def reset_filters(self) -> None:
        """Reset all filters to show all data."""
        if len(self._filtered_rows) != len(self._all_rows):
            self._filtered_rows = self._all_rows.copy()
            self._update_all_projections()
            self._broadcast_filter_change("filter_reset", ["temporal", "geo"])
            logger.info("Reset all filters - all points now visible")

    def get_summary(self) -> dict:
        """Get a summary of the current session state."""
        if not self.has_data():
            return {"status": "empty", "message": "No data loaded"}

        summary = {
            "status": "loaded",
            "shape": self._metadata["shape"],
            "columns": self._metadata["columns"],
            "memory_usage": f"{self._all_rows.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "all_rows_count": len(self._all_rows),
            "filtered_rows_count": len(self._filtered_rows),
            "filter_ratio": len(self._filtered_rows) / len(self._all_rows) if len(self._all_rows) > 0 else 0,
        }

        # Add projection state info
        summary["temporal_projection"] = self._temporal_projection.get_summary()
        summary["geo_projection"] = self._geo_projection.get_summary()

        return summary

    def get_spatial_aggregation(self, max_groups: int) -> pd.DataFrame:
        """
        Get spatially aggregated data for visualization.

        Args:
            max_groups: Maximum number of groups to return (updates projection max_rows)

        Returns:
            Aggregated DataFrame suitable for heatmap visualization
        """
        # Update the projection's max_rows and refresh if needed
        if self._geo_projection.max_rows != max_groups:
            self._geo_projection.max_rows = max_groups
            self._geo_projection.update_projection(self._filtered_rows)
        
        return self._geo_projection.projection_df

    def get_temporal_aggregation(self, max_groups: int) -> pd.DataFrame:
        """
        Get temporally aggregated data for CDF visualization.

        Args:
            max_groups: Maximum number of groups to return (updates projection max_rows)

        Returns:
            Aggregated DataFrame suitable for CDF visualization
        """
        # Update the projection's max_rows and refresh if needed
        if self._temporal_projection.max_rows != max_groups:
            self._temporal_projection.max_rows = max_groups
            self._temporal_projection.update_projection(self._filtered_rows)
        
        return self._temporal_projection.projection_df

    @property
    def filter_version(self) -> int:
        """Get the current filter version for change tracking."""
        return self._filter_version

    def _broadcast_filter_change(self, event_type: str, affected_components: list[str]) -> None:
        """Broadcast filter change event to all SSE clients."""
        self._filter_version += 1
        event_data = {
            "type": event_type,
            "affected_components": affected_components,
            "version": self._filter_version,
            "timestamp": time.time(),
            "session_state": {
                "has_data": self.has_data(),
                "all_rows_count": len(self._all_rows) if self.has_data() else 0,
                "filtered_rows_count": len(self._filtered_rows) if self.has_data() else 0,
                "columns": list(self._all_rows.columns) if self.has_data() else []
            }
        }
        
        # Queue the event for all connected SSE clients
        for client_queue in self._sse_clients:
            try:
                client_queue.put_nowait(event_data)
            except asyncio.QueueFull:
                logger.warning("SSE client queue full, dropping event")

    async def filter_change_stream(self) -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events stream for filter changes."""
        client_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_clients.add(client_queue)
        
        try:
            # Send initial connection event
            initial_event = {
                "type": "connection_established",
                "affected_components": ["temporal", "geo"],
                "version": self._filter_version,
                "timestamp": time.time(),
                "session_state": {
                    "has_data": self.has_data(),
                    "all_rows_count": len(self._all_rows) if self.has_data() else 0,
                    "filtered_rows_count": len(self._filtered_rows) if self.has_data() else 0,
                    "columns": list(self._all_rows.columns) if self.has_data() else []
                }
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
                        "version": self._filter_version
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE client disconnected")
        finally:
            self._sse_clients.discard(client_queue)