"""
Tests for compute_embeddings_cli.py.
"""

import logging
from pathlib import Path

from syrupy import SnapshotAssertion

logger = logging.getLogger(__name__)


def test_compute_embeddings_cli_happy_path(source_tree_root: Path, tmp_path: Path, snapshot: SnapshotAssertion) -> None:
    """Runs the CLI pointing at the images in a directory, verifies that embeddings and UMAP projections are correct."""
