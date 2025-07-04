"""Tests for geo plot module."""

from pathlib import Path

import pandas as pd
import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension

from crossfilter.core.schema import SchemaColumns, load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.geo_plot import create_geo_plot


class HTMLSnapshotExtension(SingleFileSnapshotExtension):
    """Custom syrupy extension to save HTML files with .html extension."""

    _file_extension = "html"

    def serialize(self, data, **kwargs) -> bytes:
        """Serialize string data to bytes for file storage."""
        if isinstance(data, str):
            return data.encode("utf-8")
        return super().serialize(data, **kwargs)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load sample data for testing."""
    sample_path = Path(__file__).parent.parent.parent / "test_data" / "sample_100.jsonl"
    return load_jsonl_to_dataframe(sample_path)


@pytest.fixture
def session_state_with_data(sample_data: pd.DataFrame) -> SessionState:
    """Create session state with sample data loaded."""
    session_state = SessionState()
    session_state.load_dataframe(sample_data)
    return session_state


@pytest.fixture
def individual_geo_data(sample_data: pd.DataFrame) -> pd.DataFrame:
    individual_df = sample_data.copy()
    return individual_df


@pytest.fixture
def grouped_geo_data(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Create grouped geo data with COUNT column."""
    grouped_df = sample_data.copy()
    # Add some COUNT values for testing marker sizing
    count_values = [1, 5, 10, 20, 50] * (len(grouped_df) // 5 + 1)
    grouped_df[SchemaColumns.COUNT] = count_values[: len(grouped_df)]
    return grouped_df


@pytest.fixture
def nyc_area_data() -> pd.DataFrame:
    """Create test data with points around NYC area to test auto-fitting."""
    # NYC coordinates: 40.7128° N, 74.0060° W
    import datetime

    nyc_lat, nyc_lon = 40.7128, -74.0060

    # Create points in a small area around NYC
    data = []
    locations = [
        ("Central Park", nyc_lat + 0.02, nyc_lon + 0.01),
        ("Brooklyn Bridge", nyc_lat - 0.01, nyc_lon + 0.005),
        ("Times Square", nyc_lat + 0.005, nyc_lon - 0.005),
        ("Statue of Liberty", nyc_lat - 0.03, nyc_lon - 0.02),
        ("One World Trade", nyc_lat - 0.005, nyc_lon + 0.002),
        ("Empire State Building", nyc_lat + 0.008, nyc_lon - 0.008),
    ]

    for i, (name, lat, lon) in enumerate(locations):
        data.append(
            {
                SchemaColumns.UUID_STRING: f"nyc_uuid_{i}",
                SchemaColumns.DATA_TYPE: "PHOTO" if i % 2 == 0 else "VIDEO",
                SchemaColumns.NAME: name,
                SchemaColumns.CAPTION: f"Photo/Video at {name}",
                SchemaColumns.SOURCE_FILE: f"nyc_file_{i}.jpg",
                SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE: "2024-01-15T12:00:00",
                SchemaColumns.TIMESTAMP_UTC: datetime.datetime(
                    2024, 1, 15, 12, i, 0, tzinfo=datetime.timezone.utc
                ),
                SchemaColumns.GPS_LATITUDE: lat,
                SchemaColumns.GPS_LONGITUDE: lon,
                SchemaColumns.RATING_0_TO_5: 4,
                SchemaColumns.SIZE_IN_BYTES: 1024000,
                SchemaColumns.COUNT: 1 + i * 2,  # Varying count values
            }
        )

    df = pd.DataFrame(data)
    df.index.name = SchemaColumns.DF_ID
    return df


def test_geo_plot_no_data(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data.head(0), title="Test Geographic Plot - No Data"
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_individual_data(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data, title="Test Geographic Plot - Individual"
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_individual_data_grouped(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data,
        groupby=SchemaColumns.DATA_TYPE,
        title="Test Geographic Plot - Individual, Grouped By Type",
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_with_count_data(
    snapshot: SnapshotAssertion, grouped_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        grouped_geo_data,
        title="Test Geographic Plot - With Count Data",
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_with_count_data_grouped(
    snapshot: SnapshotAssertion, grouped_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        grouped_geo_data,
        groupby=SchemaColumns.DATA_TYPE,
        title="Test Geographic Plot - With Count Data, Grouped By Type",
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_nyc_area_auto_fit(
    snapshot: SnapshotAssertion, nyc_area_data: pd.DataFrame
) -> None:
    """Test that plot auto-fits to NYC area data, demonstrating zoom behavior."""
    fig = create_geo_plot(
        nyc_area_data,
        groupby=SchemaColumns.DATA_TYPE,
        title="Test Geographic Plot - NYC Area Auto-Fit",
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
