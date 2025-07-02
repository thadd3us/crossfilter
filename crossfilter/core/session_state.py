"""Session state management for single-session Crossfilter application."""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import pandas as pd

from crossfilter.core.bucketing import (
    H3_LEVELS,
    BucketKey,
    add_bucketed_columns,
    bucket_by_target_column,
    get_optimal_h3_level,
    get_optimal_temporal_level,
)
from crossfilter.core.filter_state import FilterState
from crossfilter.core.schema import (
    BucketingType,
    SchemaColumns,
    TemporalLevel,
    get_h3_column_name,
    get_temporal_column_name,
)

logger = logging.getLogger(__name__)


class SessionState:
    """
    Manages the current state of data loaded into the Crossfilter application.

    This class represents the single session state for the application, holding
    the currently loaded dataset and any derived data structures needed for
    crossfiltering and visualization.

    In the single-session design pattern, there is exactly one instance of this
    class per web server instance, simplifying state management and eliminating
    the need for complex multi-user session handling.
    """

    def __init__(self) -> None:
        """Initialize session state with empty DataFrame."""
        self._data = pd.DataFrame()
        self._filter_state = FilterState()
        # Track bucketing configurations presented to the frontend
        self._bucket_keys: dict[BucketingType, BucketKey] = {}
        self._bucketed_dataframes: dict[BucketingType, pd.DataFrame] = {}
        self._update_metadata()
        
        # SSE event broadcasting
        self._filter_version = 0
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._sse_clients: set[asyncio.Queue] = set()

    @property
    def data(self) -> pd.DataFrame:
        """Get the current dataset."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        """Set the current dataset."""
        self._data = value
        bucketed_data = add_bucketed_columns(value)
        self._filter_state.initialize_with_data(bucketed_data)
        self._update_metadata()
        logger.info(f"Loaded dataset with {len(value)} rows into session state")
        
        # Broadcast data loaded event
        self._broadcast_filter_change("data_loaded", ["temporal", "spatial"])

    def _update_metadata(self) -> None:
        """Update metadata based on current DataFrame."""
        self._metadata = {
            "shape": self._data.shape,
            "columns": list(self._data.columns),
            "dtypes": self._data.dtypes.to_dict(),
        }

    @property
    def metadata(self) -> dict:
        """Get metadata about the current dataset."""
        return self._metadata.copy()

    def has_data(self) -> bool:
        """Check if the session has data loaded."""
        return not self._data.empty

    def clear(self) -> None:
        """Clear all data from the session state."""
        self._data = pd.DataFrame()
        self._filter_state = FilterState()
        self._bucket_keys = {}
        self._bucketed_dataframes = {}
        self._update_metadata()

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the session state."""
        self.data = df

    def get_summary(self) -> dict:
        """Get a summary of the current session state."""
        if not self.has_data():
            return {"status": "empty", "message": "No data loaded"}

        summary = {
            "status": "loaded",
            "shape": self._metadata["shape"],
            "columns": self._metadata["columns"],
            "memory_usage": f"{self._data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        }

        # Add filter state info
        summary["filter_state"] = self._filter_state.get_summary()

        return summary


    @property
    def filter_state(self) -> FilterState:
        """Get the filter state manager."""
        return self._filter_state

    def get_filtered_data(self) -> pd.DataFrame:
        """Get the currently filtered dataset."""
        bucketed_data = add_bucketed_columns(self._data)
        return self._filter_state.get_filtered_dataframe(bucketed_data)

    def get_spatial_aggregation(self, max_groups: int) -> pd.DataFrame:
        """
        Get spatially aggregated data for visualization.

        Args:
            max_groups: Maximum number of groups to return

        Returns:
            Aggregated DataFrame suitable for heatmap visualization
        """
        filtered_data = self.get_filtered_data()

        if len(filtered_data) <= max_groups:
            # Return individual points if under threshold
            columns = [SchemaColumns.GPS_LATITUDE, SchemaColumns.GPS_LONGITUDE]
            result = filtered_data[columns].copy()
            # Add df_id column for consistency
            result[SchemaColumns.DF_ID] = filtered_data.index
            return result

        # Find optimal H3 level
        optimal_level = get_optimal_h3_level(filtered_data, max_groups)

        if optimal_level is None:
            # Fallback to least granular level
            optimal_level = min(H3_LEVELS)

        # Aggregate by H3 cells
        h3_column = get_h3_column_name(optimal_level)
        bucketed = bucket_by_target_column(filtered_data, h3_column)

        # Rename COUNT column to count for consistency
        result = bucketed.rename(columns={"COUNT": "count"})
        return result

    def get_temporal_aggregation(self, max_groups: int) -> pd.DataFrame:
        """
        Get temporally aggregated data for CDF visualization.

        Args:
            max_groups: Maximum number of groups to return

        Returns:
            Aggregated DataFrame suitable for CDF visualization
        """
        filtered_data = self.get_filtered_data()

        if len(filtered_data) <= max_groups:
            # Return individual points if under threshold
            df = filtered_data[[SchemaColumns.TIMESTAMP_UTC]].copy()
            df = df.sort_values(SchemaColumns.TIMESTAMP_UTC)
            df["cumulative_count"] = range(1, len(df) + 1)
            # Add df_id column for consistency
            df[SchemaColumns.DF_ID] = df.index
            return df

        # Find optimal temporal level
        optimal_level = get_optimal_temporal_level(filtered_data, max_groups)

        if optimal_level is None:
            # Fallback to least granular level
            optimal_level = TemporalLevel.YEAR

        # Aggregate by temporal buckets
        temporal_column = get_temporal_column_name(optimal_level)
        bucketed = bucket_by_target_column(filtered_data, temporal_column)

        # Transform to match expected output format for CDF visualization
        result = bucketed.rename(columns={"COUNT": "count"})

        # Add df_id column for consistency with individual points
        result[SchemaColumns.DF_ID] = result.index

        # Sort by timestamp and add cumulative count for CDF
        result = result.sort_values(temporal_column)
        result["cumulative_count"] = result["count"].cumsum()

        return result

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
                "row_count": len(self._data) if self.has_data() else 0,
                "filtered_count": self._filter_state.filter_count if self.has_data() else 0,
                "columns": list(self._data.columns) if self.has_data() else []
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
                "affected_components": ["temporal", "spatial"],
                "version": self._filter_version,
                "timestamp": time.time(),
                "session_state": {
                    "has_data": self.has_data(),
                    "row_count": len(self._data) if self.has_data() else 0,
                    "filtered_count": self._filter_state.filter_count if self.has_data() else 0,
                    "columns": list(self._data.columns) if self.has_data() else []
                }
            }
            yield f"data: {json.dumps(initial_event)}\n\n"
            
            # Stream subsequent events
            while True:
                try:
                    event_data = await asyncio.wait_for(client_queue.get(), timeout=30.0)
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

