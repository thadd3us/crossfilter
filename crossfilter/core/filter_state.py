"""Filter state management with undo stack for crossfilter operations."""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Any
import pandas as pd
import logging

from crossfilter.core.schema_constants import FilterOperationType, DF_ID_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class FilterOperation:
    """Represents a single filter operation that can be undone."""
    operation_type: FilterOperationType
    filtered_df_ids: Set[int]  # df_ids that remain after filtering
    description: str  # Human-readable description


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
    
    def apply_spatial_filter(self, filtered_df_ids: Set[int], description: str) -> None:
        """
        Apply a spatial filter operation.
        
        Args:
            filtered_df_ids: Set of df_ids that should remain visible
            description: Description of the filter operation
        """
        self._push_current_state(FilterOperationType.SPATIAL, description)
        self._current_filtered_df_ids = filtered_df_ids & self._all_df_ids
        logger.info(f"Applied spatial filter: {len(self._current_filtered_df_ids)} points remain")
        
    def apply_temporal_filter(self, filtered_df_ids: Set[int], description: str) -> None:
        """
        Apply a temporal filter operation.
        
        Args:
            filtered_df_ids: Set of df_ids that should remain visible
            description: Description of the filter operation  
        """
        self._push_current_state(FilterOperationType.TEMPORAL, description)
        self._current_filtered_df_ids = filtered_df_ids & self._all_df_ids
        logger.info(f"Applied temporal filter: {len(self._current_filtered_df_ids)} points remain")
        
    def intersect_with_filter(self, new_filtered_df_ids: Set[int], operation_type: FilterOperationType,
                            description: str) -> None:
        """
        Apply a filter by intersecting with current filter state.
        
        Args:
            new_filtered_df_ids: New df_ids to intersect with current filter
            operation_type: Type of operation (spatial or temporal)
            description: Description of the filter operation
        """
        self._push_current_state(operation_type, description)
        if self._current_filtered_df_ids is not None:
            self._current_filtered_df_ids &= new_filtered_df_ids
        else:
            self._current_filtered_df_ids = new_filtered_df_ids & self._all_df_ids
        logger.info(f"Applied intersect filter: {len(self._current_filtered_df_ids)} points remain")
            
    def reset_filters(self) -> None:
        """Reset all filters to show all data."""
        if self._current_filtered_df_ids != self._all_df_ids:
            self._push_current_state(FilterOperationType.RESET, 'Reset all filters')
            self._current_filtered_df_ids = self._all_df_ids.copy()
            logger.info("Reset all filters - all points now visible")
            
    def undo(self) -> bool:
        """
        Undo the last filter operation.
        
        Returns:
            True if undo was successful, False if no operations to undo
        """
        if not self.can_undo:
            return False
            
        last_operation = self._undo_stack.pop()
        self._current_filtered_df_ids = last_operation.filtered_df_ids.copy()
        logger.info(f"Undid filter operation: {len(self._current_filtered_df_ids)} points now visible")
        return True
        
    def get_undo_stack_info(self) -> List[Dict[str, Any]]:
        """
        Get information about operations in the undo stack.
        
        Returns:
            List of operation descriptions
        """
        return [
            {
                'operation_type': op.operation_type,
                'description': op.description,
                'count': len(op.filtered_df_ids)
            }
            for op in reversed(self._undo_stack)
        ]
        
    def get_filtered_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a DataFrame filtered by the current filter state.
        
        Args:
            df: Input DataFrame with df_id as index
            
        Returns:
            Filtered DataFrame
        """
        if self._current_filtered_df_ids is None:
            return df.iloc[0:0]  # Empty DataFrame
            
        return df.loc[df.index.isin(self._current_filtered_df_ids)].copy()
        
    def _push_current_state(self, operation_type: FilterOperationType, description: str) -> None:
        """Push the current state onto the undo stack."""
        if self._current_filtered_df_ids is not None:
            operation = FilterOperation(
                operation_type=operation_type,
                filtered_df_ids=self._current_filtered_df_ids.copy(),
                description=description
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