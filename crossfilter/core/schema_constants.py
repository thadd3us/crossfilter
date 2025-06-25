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
    RATING_1_TO_5 = "RATING_1_TO_5"
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




# Standard DataFrame ID column name
DF_ID_COLUMN = "df_id"


# Helper functions to get quantized column names
def get_h3_column_name(level: int) -> str:
    """Get the H3 column name for a specific level (0-15)."""
    if not 0 <= level <= 15:
        raise ValueError(f"H3 level must be between 0 and 15, got {level}")
    return f"QUANTIZED_H3_L{level}"


def get_temporal_column_name(level: TemporalLevel) -> str:
    """Get the temporal quantization column name for a specific level."""
    return f"QUANTIZED_TIMESTAMP_{level}"
