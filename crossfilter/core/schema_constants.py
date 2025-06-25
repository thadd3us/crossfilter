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


class QuantizedColumns(StrEnum):
    """Quantized column name patterns."""
    H3_PREFIX = "QUANTIZED_H3_L"
    # THAD: There's a weird inconsistency here: the H3_PREFIX is a prefix, but the other columns are not.  Let's spell them all out.
    TIMESTAMP_SECOND = "QUANTIZED_TIMESTAMP_SECOND"
    TIMESTAMP_MINUTE = "QUANTIZED_TIMESTAMP_MINUTE"
    TIMESTAMP_HOUR = "QUANTIZED_TIMESTAMP_HOUR"
    TIMESTAMP_DAY = "QUANTIZED_TIMESTAMP_DAY"
    TIMESTAMP_MONTH = "QUANTIZED_TIMESTAMP_MONTH"
    TIMESTAMP_YEAR = "QUANTIZED_TIMESTAMP_YEAR"


# THAD: DRY: Move this class above the previous one, then use its values in f-strings when defining things like "QUANTIZED_TIMESTAMP_YEAR".  Make these all caps.
class TemporalLevel(StrEnum):
    """Temporal quantization levels."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


# Standard DataFrame ID column name
DF_ID_COLUMN = "df_id"