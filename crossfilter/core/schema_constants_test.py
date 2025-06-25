"""Tests for schema constants and enums."""

import pytest
from crossfilter.core.schema_constants import (
    SchemaColumns,
    FilterOperationType, 
    QuantizedColumns,
    TemporalLevel,
    DF_ID_COLUMN
)


def test_schema_columns_enum():
    """Test that SchemaColumns enum has expected values."""
    assert SchemaColumns.GPS_LATITUDE == "GPS_LATITUDE"
    assert SchemaColumns.GPS_LONGITUDE == "GPS_LONGITUDE"
    assert SchemaColumns.TIMESTAMP_UTC == "TIMESTAMP_UTC"
    assert SchemaColumns.UUID_STRING == "UUID_STRING"
    assert SchemaColumns.DATA_TYPE == "DATA_TYPE"


def test_filter_operation_type_enum():
    """Test that FilterOperationType enum has expected values."""
    assert FilterOperationType.SPATIAL == "spatial"
    assert FilterOperationType.TEMPORAL == "temporal"
    assert FilterOperationType.RESET == "reset"


def test_quantized_columns_enum():
    """Test that QuantizedColumns enum has expected values."""
    assert QuantizedColumns.H3_PREFIX == "QUANTIZED_H3_L"
    assert QuantizedColumns.TIMESTAMP_SECOND == "QUANTIZED_TIMESTAMP_SECOND"
    assert QuantizedColumns.TIMESTAMP_MINUTE == "QUANTIZED_TIMESTAMP_MINUTE"
    assert QuantizedColumns.TIMESTAMP_HOUR == "QUANTIZED_TIMESTAMP_HOUR"
    assert QuantizedColumns.TIMESTAMP_DAY == "QUANTIZED_TIMESTAMP_DAY"
    assert QuantizedColumns.TIMESTAMP_MONTH == "QUANTIZED_TIMESTAMP_MONTH"
    assert QuantizedColumns.TIMESTAMP_YEAR == "QUANTIZED_TIMESTAMP_YEAR"


def test_temporal_level_enum():
    """Test that TemporalLevel enum has expected values."""
    assert TemporalLevel.SECOND == "second"
    assert TemporalLevel.MINUTE == "minute"
    assert TemporalLevel.HOUR == "hour"
    assert TemporalLevel.DAY == "day"
    assert TemporalLevel.MONTH == "month"
    assert TemporalLevel.YEAR == "year"


def test_df_id_column_constant():
    """Test that DF_ID_COLUMN constant is defined correctly."""
    assert DF_ID_COLUMN == "df_id"


def test_h3_column_name_construction():
    """Test H3 column name construction."""
    level = 7
    expected = f"{QuantizedColumns.H3_PREFIX}{level}"
    assert expected == "QUANTIZED_H3_L7"


def test_temporal_column_name_construction():
    """Test temporal column name construction."""
    level = TemporalLevel.HOUR
    expected = f"QUANTIZED_TIMESTAMP_{level.upper()}"
    assert expected == "QUANTIZED_TIMESTAMP_HOUR"