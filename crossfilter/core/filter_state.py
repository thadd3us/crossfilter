"""Filter state management with undo stack for crossfilter operations."""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
import pandas as pd
import logging
from copy import deepcopy

from crossfilter.core.schema_constants import FilterOperationType, DF_ID_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class FilterOperation:
    """Represents a single filter operation that can be undone."""
    operation_type: FilterOperationType
    filtered_df_ids: Set[int]  # df_ids that remain after filtering
    description: str  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional filter params


class FilterState:
    """
    Manages the current filter state with undo/redo functionality.
    
    Tracks which data points are currently visible based on spatial and temporal
    filters applied through the visualization interface. Uses DataFrame index (df_id)
    for tracking rows.
    """
    
    def __init__(self, max_undo_steps: int = 50):
        """
        Initialize filter state.
        
        Args:
            max_undo_steps: Maximum number of undo operations to keep
        """
        self.max_undo_steps = max_undo_steps
        self._undo_stack: List[FilterOperation] = []
        self._current_filtered_df_ids: Optional[Set[int]] = None
        self._all_df_ids: Set[int] = set()
        
    def initialize_with_data(self, df: pd.DataFrame) -> None:
        """
        Initialize the filter state with a dataset.
        
        Args:
            df: DataFrame containing the data (with df_id as index)
        """
        self._all_df_ids = set(df.index)
        self._current_filtered_df_ids = self._all_df_ids.copy()
        self._undo_stack.clear()
        logger.info(f"Initialized filter state with {len(self._all_df_ids)} data points")
        
    @property
    def filtered_df_ids(self) -> Set[int]:
        """Get the currently filtered df_ids."""
        if self._current_filtered_df_ids is None:
            return set()
        return self._current_filtered_df_ids.copy()
    
    @property
    def all_df_ids(self) -> Set[int]:
        """Get all available df_ids in the dataset."""
        return self._all_df_ids.copy()
    
    @property
    def filter_count(self) -> int:
        """Get the number of currently filtered items."""
        return len(self._current_filtered_df_ids) if self._current_filtered_df_ids else 0
    
    @property
    def total_count(self) -> int:
        """Get the total number of items in the dataset."""
        return len(self._all_df_ids)
    
    @property
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self._undo_stack) > 0
    
    def apply_spatial_filter(self, filtered_uuids: Set[int], description: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Apply a spatial filter operation.
        
        Args:
            filtered_uuids: Set of UUIDs that should remain visible
            description: Description of the filter operation
            metadata: Additional metadata about the filter
        """
        self._push_current_state('spatial', description, metadata or {})
        self._current_filtered_uuids = filtered_uuids & self._all_uuids
        
    def apply_temporal_filter(self, filtered_uuids: Set[int], description: str,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Apply a temporal filter operation.
        
        Args:
            filtered_uuids: Set of UUIDs that should remain visible
            description: Description of the filter operation  
            metadata: Additional metadata about the filter
        """
        self._push_current_state('temporal', description, metadata or {})
        self._current_filtered_uuids = filtered_uuids & self._all_uuids
        
    def intersect_with_filter(self, new_filtered_uuids: Set[int], operation_type: str,
                            description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Apply a filter by intersecting with current filter state.
        
        Args:
            new_filtered_uuids: New UUIDs to intersect with current filter
            operation_type: Type of operation ('spatial' or 'temporal')
            description: Description of the filter operation
            metadata: Additional metadata about the filter
        """
        self._push_current_state(operation_type, description, metadata or {})
        if self._current_filtered_uuids is not None:
            self._current_filtered_uuids &= new_filtered_uuids
        else:
            self._current_filtered_uuids = new_filtered_uuids & self._all_uuids
            
    def reset_filters(self) -> None:
        """Reset all filters to show all data."""
        if self._current_filtered_uuids != self._all_uuids:
            self._push_current_state('reset', 'Reset all filters')
            self._current_filtered_uuids = self._all_uuids.copy()
            
    def undo(self) -> bool:
        """
        Undo the last filter operation.
        
        Returns:
            True if undo was successful, False if no operations to undo
        """
        if not self.can_undo:
            return False
            
        last_operation = self._undo_stack.pop()
        self._current_filtered_uuids = last_operation.filtered_uuids.copy()
        return True
        
    def get_undo_stack_info(self) -> List[Dict[str, Any]]:
        """
        Get information about operations in the undo stack.
        
        Returns:
            List of operation descriptions and metadata
        """
        return [
            {
                'operation_type': op.operation_type,
                'description': op.description,
                'count': len(op.filtered_uuids),
                'metadata': op.metadata
            }
            for op in reversed(self._undo_stack)
        ]
        
    def get_filtered_dataframe(self, df: pd.DataFrame, uuid_col: str = 'UUID_LONG') -> pd.DataFrame:
        """
        Get a DataFrame filtered by the current filter state.
        
        Args:
            df: Input DataFrame
            uuid_col: Name of the UUID column
            
        Returns:
            Filtered DataFrame
        """
        if self._current_filtered_uuids is None:
            return df.iloc[0:0]  # Empty DataFrame
            
        return df[df[uuid_col].isin(self._current_filtered_uuids)].copy()
        
    def _push_current_state(self, operation_type: str, description: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Push the current state onto the undo stack."""
        if self._current_filtered_uuids is not None:
            operation = FilterOperation(
                operation_type=operation_type,
                filtered_uuids=self._current_filtered_uuids.copy(),
                description=description,
                metadata=metadata or {}
            )
            self._undo_stack.append(operation)
            
            # Limit stack size
            if len(self._undo_stack) > self.max_undo_steps:
                self._undo_stack.pop(0)
                
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current filter state."""
        return {
            'total_count': self.total_count,
            'filtered_count': self.filter_count,
            'filter_ratio': self.filter_count / self.total_count if self.total_count > 0 else 0,
            'can_undo': self.can_undo,
            'undo_stack_size': len(self._undo_stack)
        }