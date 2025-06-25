from enum import StrEnum
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
import json
import logging

from crossfilter.core.schema_constants import SchemaColumns, DF_ID_COLUMN

logger = logging.getLogger(__name__)


class DataType(StrEnum):
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    GPX_TRACKPOINT = "GPX_TRACKPOINT"
    GPX_WAYPOINT = "GPX_WAYPOINT"


class DataSchema(pa.DataFrameModel):
    UUID_STRING: Series[str] = pa.Field(nullable=True)
    DATA_TYPE: Series[str] = pa.Field(nullable=True, isin=list(DataType.__members__.values()))
    NAME: Series[str] = pa.Field(nullable=True)
    CAPTION: Series[str] = pa.Field(nullable=True)
    SOURCE_FILE: Series[str] = pa.Field(nullable=True)
    TIMESTAMP_MAYBE_TIMEZONE_AWARE: Series[str] = pa.Field(nullable=True)
    TIMESTAMP_UTC: Series[pd.DatetimeTZDtype] = pa.Field(nullable=True, dtype_kwargs={"tz": "UTC"})
    GPS_LATITUDE: Series[float] = pa.Field(nullable=True, ge=-90, le=90)
    GPS_LONGITUDE: Series[float] = pa.Field(nullable=True, ge=-180, le=180)
    RATING_1_TO_5: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0, le=5)
    SIZE_IN_BYTES: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = True
        coerce = True


def load_jsonl_to_dataframe(jsonl_path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a DataFrame conforming to DataSchema.
    
    The JSONL file may have extra columns and may not specify all required columns.
    Missing schema columns will be created with null values, and data will be coerced to
    the correct types according to the schema. Extra columns are preserved in the output.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        DataFrame with schema columns validated, plus any extra columns from JSONL
    """
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    # Read JSONL file
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    
    # THAD: Do you really need this branch here?  Won't it work to just use the else branch even if records is empty?
    # THAD: In general, avoid branches if at all possible!
    if not records:
        # Create an empty DataFrame with the correct schema
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(records)
    
    # Ensure all required columns exist, adding missing ones with null values
    required_columns = [
        SchemaColumns.UUID_STRING, 
        SchemaColumns.DATA_TYPE,
        SchemaColumns.NAME,
        SchemaColumns.CAPTION,
        SchemaColumns.SOURCE_FILE,
        SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE,
        SchemaColumns.TIMESTAMP_UTC,
        SchemaColumns.GPS_LATITUDE,
        SchemaColumns.GPS_LONGITUDE,
        SchemaColumns.RATING_1_TO_5,
        SchemaColumns.SIZE_IN_BYTES
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # THAD: Can you get rid of the branches below?  The columns should always exist -- just map them with a function that supports missing values.
    # Convert SOURCE_FILE from Path objects to strings if needed
    if SchemaColumns.SOURCE_FILE in df.columns and not df[SchemaColumns.SOURCE_FILE].isna().all():
        df[SchemaColumns.SOURCE_FILE] = df[SchemaColumns.SOURCE_FILE].astype(str)
    
    # Convert TIMESTAMP_UTC to UTC timezone-aware datetime if it's a string
    if SchemaColumns.TIMESTAMP_UTC in df.columns and not df[SchemaColumns.TIMESTAMP_UTC].isna().all():
        df[SchemaColumns.TIMESTAMP_UTC] = pd.to_datetime(df[SchemaColumns.TIMESTAMP_UTC], utc=True)
    
    # Validate and coerce only the schema columns
    schema_df = df[required_columns].copy()
    validated_schema_df = DataSchema.validate(schema_df, lazy=True)
    
    # Combine validated schema columns with any extra columns from original data
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns:
        result_df = pd.concat([validated_schema_df, df[extra_columns]], axis=1)
    else:
        result_df = validated_schema_df
    
    # Set stable df_id index using pandas int64 index
    result_df = result_df.reset_index(drop=True)
    result_df.index.name = DF_ID_COLUMN
    
    # THAD: Prefer f-strings that use the '=' syntax, such as '{jsonl_path=}' to add more context to the log message and show empty strings better.
    # THAD: Please configure pyline to ignore f-strings passed to logger.
    logger.info(f"Loaded {len(result_df)} records from {jsonl_path}")
    
    return result_df