"""Tests for schema constants and enums."""

import pytest
from crossfilter.core.schema_constants import (
    SchemaColumns,
    FilterOperationType,
    TemporalLevel,
    DF_ID_COLUMN,
    get_h3_column_name,
    get_temporal_column_name,
)




def test_h3_column_name_construction() -> None:
    """Test H3 column name construction."""
    # Test the helper function
    assert get_h3_column_name(7) == "QUANTIZED_H3_L7"
    assert get_h3_column_name(0) == "QUANTIZED_H3_L0"
    assert get_h3_column_name(15) == "QUANTIZED_H3_L15"

    # Test that it raises for invalid levels
    with pytest.raises(ValueError):
        get_h3_column_name(-1)
    with pytest.raises(ValueError):
        get_h3_column_name(16)


def test_temporal_column_name_construction() -> None:
    """Test temporal column name construction."""
    # Test that the temporal columns are constructed using TemporalLevel enum
    assert get_temporal_column_name(TemporalLevel.HOUR) == "QUANTIZED_TIMESTAMP_HOUR"
    assert get_temporal_column_name(TemporalLevel.SECOND) == "QUANTIZED_TIMESTAMP_SECOND"
    assert get_temporal_column_name(TemporalLevel.YEAR) == "QUANTIZED_TIMESTAMP_YEAR"
