"""Tests for filter state management."""

import pytest
import pandas as pd

from crossfilter.core.filter_state import FilterState, FilterOperation
from crossfilter.core.schema_constants import FilterOperationType, DF_ID_COLUMN


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame({
        'col1': [f"value_{i}" for i in range(100)],
        'col2': [i * 10 for i in range(100)]
    })
    df.index.name = DF_ID_COLUMN
    return df


def test_filter_state_initialization() -> None:
    """Test FilterState initialization."""
    filter_state = FilterState()
    
    assert filter_state.total_count == 0
    assert filter_state.filter_count == 0
    assert not filter_state.can_undo
    assert filter_state.filtered_df_ids == set()
    assert filter_state.all_df_ids == set()


def test_initialize_with_data(sample_df) -> None:
    """Test initializing filter state with data."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    assert filter_state.total_count == 100
    assert filter_state.filter_count == 100  # All points initially visible
    assert filter_state.all_df_ids == set(range(100))
    assert filter_state.filtered_df_ids == set(range(100))
    assert not filter_state.can_undo  # No operations yet


def test_apply_spatial_filter(sample_df) -> None:
    """Test applying spatial filter."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Apply filter to first 50 points
    filtered_ids = set(range(50))
    filter_state.apply_spatial_filter(filtered_ids, "Spatial filter test")
    
    assert filter_state.filter_count == 50
    assert filter_state.filtered_df_ids == filtered_ids
    assert filter_state.can_undo
    
    # Check undo stack
    history = filter_state.get_undo_stack_info()
    assert len(history) == 1
    assert history[0]['operation_type'] == FilterOperationType.SPATIAL
    assert history[0]['description'] == "Spatial filter test"
    assert history[0]['count'] == 100  # Previous state had all points


def test_apply_temporal_filter(sample_df) -> None:
    """Test applying temporal filter."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Apply filter to last 30 points
    filtered_ids = set(range(70, 100))
    filter_state.apply_temporal_filter(filtered_ids, "Temporal filter test")
    
    assert filter_state.filter_count == 30
    assert filter_state.filtered_df_ids == filtered_ids
    assert filter_state.can_undo


def test_intersect_with_filter(sample_df) -> None:
    """Test intersecting filters."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # First filter: first 60 points
    first_filter = set(range(60))
    filter_state.apply_spatial_filter(first_filter, "First filter")
    assert filter_state.filter_count == 60
    
    # Intersect with: points 40-80
    second_filter = set(range(40, 80))
    filter_state.intersect_with_filter(
        second_filter, 
        FilterOperationType.TEMPORAL, 
        "Intersect filter"
    )
    
    # Should have intersection: 40-59 (20 points)
    expected = first_filter & second_filter
    assert filter_state.filtered_df_ids == expected
    assert filter_state.filter_count == 20
    
    # Should have 2 operations in undo stack
    history = filter_state.get_undo_stack_info()
    assert len(history) == 2


def test_reset_filters(sample_df) -> None:
    """Test resetting all filters."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Apply some filters
    filter_state.apply_spatial_filter(set(range(25)), "Reduce to 25")
    filter_state.apply_temporal_filter(set(range(10)), "Reduce to 10")
    
    assert filter_state.filter_count == 10
    assert filter_state.can_undo
    
    # Reset filters
    filter_state.reset_filters()
    
    assert filter_state.filter_count == 100  # All points visible
    assert filter_state.filtered_df_ids == filter_state.all_df_ids
    assert filter_state.can_undo  # Can undo the reset
    
    # Should have 3 operations in undo stack
    history = filter_state.get_undo_stack_info()
    assert len(history) == 3
    assert history[0]['operation_type'] == FilterOperationType.RESET


def test_undo_operations(sample_df) -> None:
    """Test undo functionality."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Apply sequence of filters
    filter_state.apply_spatial_filter(set(range(50)), "Filter to 50")
    assert filter_state.filter_count == 50
    
    filter_state.apply_temporal_filter(set(range(25)), "Filter to 25")
    assert filter_state.filter_count == 25
    
    filter_state.apply_spatial_filter(set(range(10)), "Filter to 10")
    assert filter_state.filter_count == 10
    
    # Undo last operation (back to 25)
    success = filter_state.undo()
    assert success
    assert filter_state.filter_count == 25
    
    # Undo second operation (back to 50)
    success = filter_state.undo()
    assert success
    assert filter_state.filter_count == 50
    
    # Undo first operation (back to 100)
    success = filter_state.undo()
    assert success
    assert filter_state.filter_count == 100
    
    # No more operations to undo
    success = filter_state.undo()
    assert not success
    assert not filter_state.can_undo


def test_undo_stack_limit() -> None:
    """Test that undo stack respects maximum size."""
    filter_state = FilterState(max_undo_steps=3)
    df = pd.DataFrame({'col': range(10)})
    df.index.name = DF_ID_COLUMN
    filter_state.initialize_with_data(df)
    
    # Apply 5 filters (more than max_undo_steps)
    for i in range(5):
        filtered_ids = set(range(9 - i))
        filter_state.apply_spatial_filter(filtered_ids, f"Filter {i}")
    
    # Should only have 3 operations in stack
    history = filter_state.get_undo_stack_info()
    assert len(history) == 3
    
    # Should be the last 3 operations
    assert history[0]['description'] == "Filter 4"
    assert history[1]['description'] == "Filter 3"
    assert history[2]['description'] == "Filter 2"


def test_get_filtered_dataframe(sample_df) -> None:
    """Test getting filtered DataFrame."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Test with no filter (should return all data)
    filtered = filter_state.get_filtered_dataframe(sample_df)
    assert len(filtered) == 100
    assert list(filtered.index) == list(range(100))
    
    # Apply filter and test
    filter_ids = set(range(20, 30))
    filter_state.apply_spatial_filter(filter_ids, "Test filter")
    
    filtered = filter_state.get_filtered_dataframe(sample_df)
    assert len(filtered) == 10
    assert set(filtered.index) == filter_ids
    
    # Test with empty filter
    filter_state.apply_spatial_filter(set(), "Empty filter")
    filtered = filter_state.get_filtered_dataframe(sample_df)
    assert len(filtered) == 0


def test_filter_with_invalid_df_ids(sample_df) -> None:
    """Test filtering with df_ids that don't exist in dataset."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    # Include some invalid df_ids (beyond dataset range)
    filter_ids = set(range(90, 150))  # 90-99 valid, 100-149 invalid
    filter_state.apply_spatial_filter(filter_ids, "Mixed valid/invalid IDs")
    
    # Should only keep valid df_ids
    assert filter_state.filter_count == 10  # Only 90-99
    assert filter_state.filtered_df_ids == set(range(90, 100))


def test_get_summary(sample_df) -> None:
    """Test getting filter state summary."""
    filter_state = FilterState()
    filter_state.initialize_with_data(sample_df)
    
    summary = filter_state.get_summary()
    assert summary['total_count'] == 100
    assert summary['filtered_count'] == 100
    assert summary['filter_ratio'] == 1.0
    assert not summary['can_undo']
    assert summary['undo_stack_size'] == 0
    
    # Apply filter and check summary
    filter_state.apply_spatial_filter(set(range(25)), "Quarter filter")
    
    summary = filter_state.get_summary()
    assert summary['total_count'] == 100
    assert summary['filtered_count'] == 25
    assert summary['filter_ratio'] == 0.25
    assert summary['can_undo']
    assert summary['undo_stack_size'] == 1


def test_filter_operation_dataclass() -> None:
    """Test FilterOperation dataclass."""
    df_ids = set(range(10))
    operation = FilterOperation(
        operation_type=FilterOperationType.SPATIAL,
        filtered_df_ids=df_ids,
        description="Test operation",
        metadata={"key": "value"}
    )
    
    assert operation.operation_type == FilterOperationType.SPATIAL
    assert operation.filtered_df_ids == df_ids
    assert operation.description == "Test operation"
    assert operation.metadata == {"key": "value"}


def test_multiple_resets() -> None:
    """Test that multiple resets don't add unnecessary undo entries."""
    filter_state = FilterState()
    df = pd.DataFrame({'col': range(10)})
    df.index.name = DF_ID_COLUMN
    filter_state.initialize_with_data(df)
    
    # Apply a filter
    filter_state.apply_spatial_filter(set(range(5)), "Initial filter")
    assert filter_state.can_undo
    
    # Reset multiple times
    filter_state.reset_filters()
    filter_state.reset_filters()  # Should not add another undo entry
    filter_state.reset_filters()  # Should not add another undo entry
    
    # Should only have 2 operations: initial filter + one reset
    history = filter_state.get_undo_stack_info()
    assert len(history) == 2
    assert history[0]['operation_type'] == FilterOperationType.RESET
    assert history[1]['operation_type'] == FilterOperationType.SPATIAL


def test_empty_dataframe() -> None:
    """Test filter state with empty DataFrame."""
    filter_state = FilterState()
    empty_df = pd.DataFrame()
    empty_df.index.name = DF_ID_COLUMN
    
    filter_state.initialize_with_data(empty_df)
    
    assert filter_state.total_count == 0
    assert filter_state.filter_count == 0
    assert filter_state.filtered_df_ids == set()
    
    # Operations on empty state should handle gracefully
    filter_state.apply_spatial_filter(set(), "Empty filter")
    assert filter_state.filter_count == 0
    
    filtered = filter_state.get_filtered_dataframe(empty_df)
    assert len(filtered) == 0