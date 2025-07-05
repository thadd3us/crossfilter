"""Shared schema definitions for backend-frontend communication."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectionType(StrEnum):
    """Types of data projections supported by crossfilter."""
    
    TEMPORAL = "temporal"
    GEO = "geo"
    CLIP_EMBEDDING = "clip_embedding"  # placeholder for future


class FilterOperatorType(StrEnum):
    """Types of filter operations supported by crossfilter."""
    
    INTERSECTION = "intersection"  # Limit to current selection
    SUBTRACTION = "subtraction"    # Remove current selection


@dataclass(frozen=True)
class FilterEvent:
    """Represents a filter event from a specific projection."""
    
    projection_type: ProjectionType
    selected_df_ids: set[int]
    filter_operator: FilterOperatorType


# API Request Models
class LoadDataRequest(BaseModel):
    """Request model for loading data."""
    
    file_path: str


class DfIdsFilterRequest(BaseModel):
    """Request model for filtering to specific df_ids from a plot."""
    
    df_ids: list[int] = Field(..., description="df_ids from the plot (could be from lasso selection, visible area, etc.)")
    event_source: ProjectionType = Field(..., description="Which plot type this filtering comes from")
    filter_operator: FilterOperatorType = Field(..., description="Type of filter operation to apply")


# API Response Models
class SessionStateResponse(BaseModel):
    """Response model for session state information."""
    
    all_rows_count: int
    filtered_rows_count: int
    columns: list[str]
    memory_usage_mb: str
    temporal_projection: dict[str, Any]
    geo_projection: dict[str, Any]
    # Frontend-compatible aliases
    has_data: bool
    row_count: int
    filtered_count: int


class LoadDataResponse(BaseModel):
    """Response model for data loading operations."""
    
    success: bool
    message: str
    session_state: SessionStateResponse


class FilterResponse(BaseModel):
    """Response model for filter operations."""
    
    success: bool
    filter_state: SessionStateResponse


class TemporalPlotResponse(BaseModel):
    """Response model for temporal plot data."""
    
    plotly_plot: dict[str, Any]
    data_type: str
    point_count: int
    distinct_point_count: int
    aggregation_level: Optional[str]


class GeoPlotResponse(BaseModel):
    """Response model for geo plot data."""
    
    plotly_plot: dict[str, Any]
    marker_count: int
    distinct_point_count: int
    aggregation_level: Optional[str]


# Server-Sent Events Models
class SSEEvent(BaseModel):
    """Base model for Server-Sent Events."""
    
    type: str
    timestamp: float
    version: int


class FilterChangeEvent(SSEEvent):
    """Server-Sent Event for filter changes."""
    
    affected_components: list[str]
    session_state: SessionStateResponse


class HeartbeatEvent(SSEEvent):
    """Server-Sent Event for heartbeat."""
    
    pass


class ConnectionEstablishedEvent(FilterChangeEvent):
    """Server-Sent Event for connection establishment."""
    
    pass


