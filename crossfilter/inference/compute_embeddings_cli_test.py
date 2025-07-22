"""
Tests for compute_embeddings_cli.py.
"""

import logging
import shlex
import shutil
import sqlite3
import subprocess
from pathlib import Path
import threading
from typing import TextIO

import msgpack_numpy
import numpy as np
import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core import schema
from crossfilter.core.schema import EmbeddingType, SchemaColumns as C

# Enable msgpack_numpy for deserialization
msgpack_numpy.patch()

logger = logging.getLogger(__name__)


def read_pipe_in_thread(pipe: TextIO, name: str) -> None:
    """Reads lines from a pipe and prints them with a prefix."""
    for line in iter(pipe.readline, ""):
        print(f"[{name}] {line.strip()}")
    pipe.close()


def run_and_stream_output(command: list[str], timeout: float) -> subprocess.Popen[str]:
    """Runs a command and streams its stdout and stderr in real-time."""
    logger.info(f"Running CLI command: {shlex.join(command)}")

    process: subprocess.Popen[str] = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # Line-buffered output
        universal_newlines=True,  # Decode output as text
    )

    # Create threads to read from stdout and stderr concurrently
    stdout_thread = threading.Thread(
        target=read_pipe_in_thread, args=(process.stdout, "STDOUT"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=read_pipe_in_thread, args=(process.stderr, "STDERR"), daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()

    # Wait for both threads to finish
    stdout_thread.join(timeout=timeout)
    stderr_thread.join(timeout=timeout)

    # Wait for the subprocess to terminate and get its return code
    process.wait(timeout=timeout)
    return process


@pytest.mark.resource_intensive
def test_compute_embeddings_cli_fake_embeddings(
    source_tree_root: Path, tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    test_photos_dir = source_tree_root / "test_data" / "test_photos"
    output_db = tmp_path / "embeddings.db"

    # Run the CLI with SIGLIP2 embeddings and UMAP projection
    cmd = [
        *("python", "-m"),
        "crossfilter.inference.compute_embeddings_cli",
        *("--embedding_type", str(EmbeddingType.FAKE_EMBEDDING_FOR_TESTING)),
        *("--input_dir", str(test_photos_dir)),
        *("--output_embeddings_db", str(output_db)),
        *("--batch_size", "2"),  # Small batch size for testing
        "--recompute_existing_embeddings",
        "--reproject_umap_embeddings",
    ]
    process = run_and_stream_output(cmd, timeout=5)
    assert process.returncode == 0, f"CLI failed with stderr: {process.stderr}"

    assert output_db.exists(), "Output database was not created."

    # Connect to database and verify structure
    with sqlite3.connect(output_db) as conn:
        # Check that tables were created
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        actual_tables = sorted(tables["name"])
        T = schema.EmbeddingsTables
        expected_tables = sorted({T.EMBEDDINGS, T.UMAP_MODEL})
        assert actual_tables == expected_tables

        # Load embeddings table
        embeddings_df = pd.read_sql(f"SELECT * FROM {T.EMBEDDINGS}", conn)
        # The database now uses schema constant column names directly
        assert embeddings_df.columns.tolist() == [C.UUID_STRING, C.SEMANTIC_EMBEDDING]

        umap_model_df = pd.read_sql(f"SELECT * FROM {T.UMAP_MODEL}", conn)
        assert len(umap_model_df) == 1, "Should have exactly one UMAP model"

    # Deserialize and round embeddings for consistent testing
    embeddings_df[C.SEMANTIC_EMBEDDING] = embeddings_df[C.SEMANTIC_EMBEDDING].apply(
        msgpack_numpy.loads
    )
    embeddings_df[C.SEMANTIC_EMBEDDING] = embeddings_df[C.SEMANTIC_EMBEDDING].apply(
        lambda x: np.round(x, 3)
    )
    # Sort by UUID for consistent test results
    embeddings_df = embeddings_df.sort_values(C.UUID_STRING).reset_index(drop=True)
    assert embeddings_df.to_dict(orient="records") == snapshot
