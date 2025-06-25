"""Tests for schema constants and enums."""

import pytest
from crossfilter.core.schema_constants import (
    TemporalLevel,
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
    # Test the helper function
    assert get_temporal_column_name(TemporalLevel.HOUR) == "QUANTIZED_TIMESTAMP_HOUR"
    assert get_temporal_column_name(TemporalLevel.DAY) == "QUANTIZED_TIMESTAMP_DAY"
    assert get_temporal_column_name(TemporalLevel.SECOND) == "QUANTIZED_TIMESTAMP_SECOND"