"""Schema constants and enums for crossfilter data structures."""

from enum import StrEnum


class SchemaColumns(StrEnum):
    """Column names from the DataSchema."""

    UUID_STRING = "UUID_STRING"
    DATA_TYPE = "DATA_TYPE"
    NAME = "NAME"
    CAPTION = "CAPTION"
    SOURCE_FILE = "SOURCE_FILE"
    TIMESTAMP_MAYBE_TIMEZONE_AWARE = "TIMESTAMP_MAYBE_TIMEZONE_AWARE"
    TIMESTAMP_UTC = "TIMESTAMP_UTC"
    GPS_LATITUDE = "GPS_LATITUDE"
    GPS_LONGITUDE = "GPS_LONGITUDE"
    RATING_0_TO_5 = "RATING_0_TO_5"
    SIZE_IN_BYTES = "SIZE_IN_BYTES"


class FilterOperationType(StrEnum):
    """Types of filter operations."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    RESET = "reset"


class TemporalLevel(StrEnum):
    """Temporal quantization levels."""

    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"


# THAD: Actually, let's get rid of this enum and instead just offer a function that can yield these strings if necessary.  We shouldn't need to identify these individually in code.
class QuantizedColumns(StrEnum):
    """Quantized column name patterns."""

    # H3 hex grid columns at different levels
    QUANTIZED_H3_L0 = "QUANTIZED_H3_L0"
    QUANTIZED_H3_L1 = "QUANTIZED_H3_L1"
    QUANTIZED_H3_L2 = "QUANTIZED_H3_L2"
    QUANTIZED_H3_L3 = "QUANTIZED_H3_L3"
    QUANTIZED_H3_L4 = "QUANTIZED_H3_L4"
    QUANTIZED_H3_L5 = "QUANTIZED_H3_L5"
    QUANTIZED_H3_L6 = "QUANTIZED_H3_L6"
    QUANTIZED_H3_L7 = "QUANTIZED_H3_L7"
    QUANTIZED_H3_L8 = "QUANTIZED_H3_L8"
    QUANTIZED_H3_L9 = "QUANTIZED_H3_L9"
    QUANTIZED_H3_L10 = "QUANTIZED_H3_L10"
    QUANTIZED_H3_L11 = "QUANTIZED_H3_L11"
    QUANTIZED_H3_L12 = "QUANTIZED_H3_L12"
    QUANTIZED_H3_L13 = "QUANTIZED_H3_L13"
    QUANTIZED_H3_L14 = "QUANTIZED_H3_L14"
    QUANTIZED_H3_L15 = "QUANTIZED_H3_L15"
    # Temporal quantization columns using TemporalLevel enum values
    QUANTIZED_TIMESTAMP_SECOND = f"QUANTIZED_TIMESTAMP_{TemporalLevel.SECOND}"
    QUANTIZED_TIMESTAMP_MINUTE = f"QUANTIZED_TIMESTAMP_{TemporalLevel.MINUTE}"
    QUANTIZED_TIMESTAMP_HOUR = f"QUANTIZED_TIMESTAMP_{TemporalLevel.HOUR}"
    QUANTIZED_TIMESTAMP_DAY = f"QUANTIZED_TIMESTAMP_{TemporalLevel.DAY}"
    QUANTIZED_TIMESTAMP_MONTH = f"QUANTIZED_TIMESTAMP_{TemporalLevel.MONTH}"
    QUANTIZED_TIMESTAMP_YEAR = f"QUANTIZED_TIMESTAMP_{TemporalLevel.YEAR}"


# Standard DataFrame ID column name
DF_ID_COLUMN = "df_id"


# Helper function to get H3 column name for a specific level
def get_h3_column_name(level: int) -> str:
    """Get the H3 column name for a specific level (0-15)."""
    if not 0 <= level <= 15:
        raise ValueError(f"H3 level must be between 0 and 15, got {level}")
    return f"QUANTIZED_H3_L{level}"
