"""Tests for fake embedding functions."""

import logging
from decimal import Decimal

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.schema import SchemaColumns as C
from crossfilter.inference.fake_embedding_functions import (
    FakeEmbedder,
)
from crossfilter.inference.test_fixtures import test_df

assert test_df is not None, "Preserve import."

logger = logging.getLogger(__name__)


@pytest.fixture
def embedder() -> FakeEmbedder:
    """Create FakeEmbedder instance for testing."""
    return FakeEmbedder()


def test_compute_image_embeddings(
    test_df: pd.DataFrame, embedder: FakeEmbedder, snapshot: SnapshotAssertion
) -> None:
    embedder.compute_image_embeddings(test_df, C.SOURCE_FILE, C.SEMANTIC_EMBEDDING)
    test_df = test_df.drop(columns=[C.SOURCE_FILE])
    test_df[C.SEMANTIC_EMBEDDING] = test_df[C.SEMANTIC_EMBEDDING].apply(
        lambda x: (
            [float(Decimal(str(val)).quantize(Decimal("0.0001"))) for val in x]
            if x is not None
            else None
        )
    )
    assert test_df.to_dict(orient="records") == snapshot


def test_compute_text_embeddings(
    test_df: pd.DataFrame, embedder: FakeEmbedder, snapshot: SnapshotAssertion
) -> None:
    embedder.compute_text_embeddings(test_df, C.CAPTION, C.SEMANTIC_EMBEDDING)
    test_df = test_df.drop(columns=[C.SOURCE_FILE])
    test_df[C.SEMANTIC_EMBEDDING] = test_df[C.SEMANTIC_EMBEDDING].apply(
        lambda x: (
            [float(Decimal(str(val)).quantize(Decimal("0.0001"))) for val in x]
            if x is not None
            else None
        )
    )
    assert test_df.to_dict(orient="records") == snapshot
