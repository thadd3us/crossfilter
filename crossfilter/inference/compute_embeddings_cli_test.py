"""
Tests for compute_embeddings_cli.py.
"""

import logging
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from syrupy import SnapshotAssertion
from typer.testing import CliRunner

from crossfilter.core.schema import EmbeddingType, SchemaColumns
from crossfilter.inference.compute_embeddings_cli import (
    app,
    main,
)

logger = logging.getLogger(__name__)


def test_compute_embeddings_cli_happy_path(source_tree_root: Path, tmp_path: Path, snapshot: SnapshotAssertion) -> None:
    """Runs the CLI pointing at the images in a directory, verifies that embeddings and UMAP projections are correct."""
    