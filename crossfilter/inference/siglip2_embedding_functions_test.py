"""
Tests for SigLIP2 embedding functions.
"""

import logging
from pathlib import Path
import pandas as pd

import numpy as np
import pytest
from syrupy import SnapshotAssertion
from torch import cosine_similarity
from crossfilter.core.schema import SchemaColumns as C

import scipy.spatial.distance
import plotly.express as px

from crossfilter.inference.siglip2_embedding_functions import (
    SigLIP2Embedder,
)
from tests.util.syrupy_html_snapshot import HTMLSnapshotExtension

# Mark all tests in this file as resource intensive to avoid downloading the SIGLIP2 model in regular test runs
# Use fake_embedding_functions_test.py for fast testing instead
# To run these tests: pytest -m "resource_intensive"
pytestmark = pytest.mark.resource_intensive

logger = logging.getLogger(__name__)


@pytest.fixture
def embedder() -> SigLIP2Embedder:
    """Create SigLIP2Embedder instance for testing."""
    return SigLIP2Embedder()


@pytest.fixture
def test_df(source_tree_root: Path) -> pd.DataFrame:

    filenames = """00_munich_rathaus.jpg
01_golden_gate_bridge.jpg
02_earth_from_space.jpg
03_backlit_man_looking_out.jpg
04_mountain_view.jpg
05_astronaut_on_moon.jpg
06_fireworks_on_blue_sunset.jpg
07_paper_bundle_and_pen.jpg
08_herman_hesse.jpg
09_martin_luther_king_jr.jpg""".split(
        "\n"
    )
    df = pd.DataFrame({"filename": filenames})
    df[C.SOURCE_FILE] = df["filename"].map(
        lambda x: source_tree_root / "test_data" / "test_photos" / x
    )
    df[C.CAPTION] = [
        "Minga Rathaus an einem schönen sonnigen Tag mit blauem Himmel",
        "Golden Gate Bridge in San Francisco",
        "Planet Earth as seen from space showing blue oceans and white clouds",
        "A backlit man looking out over a gray scene",
        "Mountain peaks, in black and white",
        "Astronaut in a spacesuit on the gray lunar surface",
        "Fireworks exploding in front of a deep blue sunset",
        "Aged bundle of papers tied with twine, and an old-fashioned pen",
        "Herman Hesse, der ein Buch liest und eine Brille trägt",
        "Martin Luther King, Jr.",
    ]
    assert df[C.SOURCE_FILE].map(Path.exists).all()
    return df


def test_compute_image_and_text_embeddings_match(
    test_df: pd.DataFrame, embedder: SigLIP2Embedder, snapshot: SnapshotAssertion
) -> None:
    df = test_df
    df["image_embedding"] = embedder.compute_image_embeddings(df[C.SOURCE_FILE].to_list())
    # df[C.CAPTION] = ("A photo of " + df[C.CAPTION]).str.lower()
    df["text_embedding"] = embedder.compute_text_embeddings(df[C.CAPTION].to_list())

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


def test_error_handling_missing_image(embedder: SigLIP2Embedder) -> None:
    """Test error handling for missing image files."""
    missing_path = Path("/nonexistent/image.jpg")

    with pytest.raises(FileNotFoundError, match="Image not found"):
        embedder.compute_image_embeddings([missing_path])


def test_error_handling_invalid_image(embedder: SigLIP2Embedder, tmp_path: Path) -> None:
    """Test error handling for invalid image files."""
    # Create a non-image file
    invalid_file = tmp_path / "not_an_image.jpg"
    invalid_file.write_text("This is not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        embedder.compute_image_embeddings([invalid_file])


def test_empty_inputs(embedder: SigLIP2Embedder) -> None:
    """Test handling of empty input lists."""
    # Empty image list
    image_embeddings = embedder.compute_image_embeddings([])
    assert image_embeddings == []

    # Empty text list
    text_embeddings = embedder.compute_text_embeddings([])
    assert text_embeddings == []

    # # Empty embeddings for caption generation
    # captions = generate_captions_from_image_embeddings([])
    # assert captions == []


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
