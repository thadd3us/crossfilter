"""
Tests for SigLIP2 embedding functions.
"""

import logging
from pathlib import Path
import pandas as pd

import numpy as np
import pytest
from syrupy import SnapshotAssertion
from crossfilter.core.schema import SchemaColumns as C

import scipy.spatial.distance
import plotly.express as px

from crossfilter.inference.siglip2_embedding_functions import (
    SigLIP2Embedder,
)
from tests.util.syrupy_html_snapshot import HTMLSnapshotExtension
from crossfilter.inference.test_fixtures import test_df

# Mark all tests in this file as resource intensive to avoid downloading the SIGLIP2 model in regular test runs
# Use fake_embedding_functions_test.py for fast testing instead
# To run these tests: pytest -m "resource_intensive"
pytestmark = pytest.mark.resource_intensive

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def embedder() -> SigLIP2Embedder:
    """Create SigLIP2Embedder instance for testing."""
    return SigLIP2Embedder()


def test_compute_image_and_text_embeddings_match(
    test_df: pd.DataFrame, embedder: SigLIP2Embedder, snapshot: SnapshotAssertion
) -> None:
    df = test_df
    embedder.compute_image_embeddings(df, C.SOURCE_FILE, "image_embedding")
    # df[C.CAPTION] = ("A photo of " + df[C.CAPTION]).str.lower()
    embedder.compute_text_embeddings(df, C.CAPTION, "text_embedding")

    distances = scipy.spatial.distance.cdist(
        np.stack(df["image_embedding"].values),
        np.stack(df["text_embedding"].values),
        metric="cosine",
    )
    assert distances.shape == (len(df), len(df))
    distances_df = pd.DataFrame(distances, index=df["filename"], columns=df[C.CAPTION])
    fig = px.imshow(distances_df)
    html_content = fig.to_html(
        div_id="test-plot-div",
        include_plotlyjs="cdn",
    )
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)

    df["image_embedding"] = df["image_embedding"].map(lambda x: np.round(x, 4))
    df["text_embedding"] = df["text_embedding"].map(lambda x: np.round(x, 4))
    assert (
        df[[C.CAPTION, "image_embedding", "text_embedding"]].to_dict(orient="records")
        == snapshot
    )


def test_error_handling_invalid_image(
    test_df: pd.DataFrame,
    embedder: SigLIP2Embedder,
    tmp_path: Path,
    snapshot: SnapshotAssertion,
) -> None:
    """Test error handling for invalid image files."""
    # Create a non-image file
    invalid_file = tmp_path / "not_an_image.jpg"
    invalid_file.write_text("This is not an image")

    test_df = test_df.head(3).copy()
    test_df.loc[1, C.SOURCE_FILE] = invalid_file

    embedder.compute_image_embeddings(test_df, "image_path", "embedding")
    assert test_df[[C.CAPTION, "image_embedding"]].to_dict(orient="records") == snapshot


def test_empty_inputs(embedder: SigLIP2Embedder) -> None:
    """Test handling of empty input DataFrames."""
    # Empty image DataFrame
    empty_df = pd.DataFrame({"image_path": []})
    with pytest.raises(ValueError):
        embedder.compute_image_embeddings(empty_df, "image_path", "embedding")
    assert empty_df.empty  # DataFrame should still be empty

    # Empty text DataFrame
    empty_df = pd.DataFrame({"caption": []})
    with pytest.raises(ValueError):
        embedder.compute_text_embeddings(empty_df, "caption", "embedding")
    assert empty_df.empty  # DataFrame should still be empty


# def test_generate_captions_from_image_embeddings(
#     test_df: pd.DataFrame, snapshot: SnapshotAssertion
# ) -> None:
#     """Test caption generation from image embeddings."""
#     df = test_df
#     image_embeddings = compute_image_embeddings(
#         df[C.SOURCE_FILE].to_list(), batch_size=8
#     )
#     df["generated_caption"] = generate_captions_from_image_embeddings(
#         image_embeddings, batch_size=4
#     )
#     assert df[["filename", "generated_caption"]].to_dict(orient="records") == snapshot
