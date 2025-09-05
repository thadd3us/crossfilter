"""Tests for Google Takeout CLI ingestion."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from syrupy.assertion import SnapshotAssertion

from crossfilter.data_ingestion.gpx.google_takeout_parser_test import create_sample_takeout_data


def test_ingest_google_takeout_cli(tmp_path: Path, snapshot: SnapshotAssertion) -> None:
    """Test CLI ingestion of Google Takeout Records.json files."""
    takeout_dir = tmp_path / "takeout_data"
    takeout_dir.mkdir()
    
    records_file = takeout_dir / "Records.json"
    with open(records_file, "w") as f:
        json.dump(create_sample_takeout_data(), f)
    
    output_parquet = tmp_path / "output.parquet"
    
    # Run the CLI via subprocess
    result = subprocess.run([
        sys.executable, "-m", "crossfilter.data_ingestion.gpx.ingest_google_takeout",
        str(takeout_dir), str(output_parquet)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert output_parquet.exists()
    
    df = pd.read_parquet(output_parquet)
    assert df.to_dict(orient="records") == snapshot