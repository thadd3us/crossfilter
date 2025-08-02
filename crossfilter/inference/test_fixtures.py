from pathlib import Path

import pandas as pd
import pytest

from crossfilter.core.schema import SchemaColumns as C


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
