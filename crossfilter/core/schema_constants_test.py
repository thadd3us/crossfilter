"""Tests for schema constants and enums."""

import pytest
from crossfilter.core.schema_constants import (
    SchemaColumns,
    FilterOperationType, 
    QuantizedColumns,
    TemporalLevel,
    DF_ID_COLUMN,
    get_h3_column_name
)


def test_schema_columns_enum() -> None:
    """Test that SchemaColumns enum has expected values."""
    assert SchemaColumns.GPS_LATITUDE == "GPS_LATITUDE"
    assert SchemaColumns.GPS_LONGITUDE == "GPS_LONGITUDE"
    assert SchemaColumns.TIMESTAMP_UTC == "TIMESTAMP_UTC"
    assert SchemaColumns.UUID_STRING == "UUID_STRING"
    assert SchemaColumns.DATA_TYPE == "DATA_TYPE"


def test_filter_operation_type_enum() -> None:
    """Test that FilterOperationType enum has expected values."""
    assert FilterOperationType.SPATIAL == "spatial"
    assert FilterOperationType.TEMPORAL == "temporal"
    assert FilterOperationType.RESET == "reset"


def test_quantized_columns_enum() -> None:
    """Test that QuantizedColumns enum has expected values."""
    # Test specific H3 levels
    assert QuantizedColumns.QUANTIZED_H3_L0 == "QUANTIZED_H3_L0"
    assert QuantizedColumns.QUANTIZED_H3_L7 == "QUANTIZED_H3_L7"
    assert QuantizedColumns.QUANTIZED_H3_L15 == "QUANTIZED_H3_L15"
    
    # Test temporal columns with new DRY approach
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_SECOND == "QUANTIZED_TIMESTAMP_SECOND"
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_MINUTE == "QUANTIZED_TIMESTAMP_MINUTE"
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_HOUR == "QUANTIZED_TIMESTAMP_HOUR"
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_DAY == "QUANTIZED_TIMESTAMP_DAY"
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_MONTH == "QUANTIZED_TIMESTAMP_MONTH"
    assert QuantizedColumns.QUANTIZED_TIMESTAMP_YEAR == "QUANTIZED_TIMESTAMP_YEAR"


def test_temporal_level_enum() -> None:
    """Test that TemporalLevel enum has expected values."""
    assert TemporalLevel.SECOND == "SECOND"
    assert TemporalLevel.MINUTE == "MINUTE"
    assert TemporalLevel.HOUR == "HOUR"
    assert TemporalLevel.DAY == "DAY"
    assert TemporalLevel.MONTH == "MONTH"
    assert TemporalLevel.YEAR == "YEAR"


def test_df_id_column_constant() -> None:
    """Test that DF_ID_COLUMN constant is defined correctly."""
    assert DF_ID_COLUMN == "df_id"


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
    level = TemporalLevel.HOUR
    expected = f"QUANTIZED_TIMESTAMP_{level}"
    assert expected == "QUANTIZED_TIMESTAMP_HOUR"
    assert expected == QuantizedColumns.QUANTIZED_TIMESTAMP_HOUR