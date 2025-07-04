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
    grouped_df[SchemaColumns.COUNT] = count_values[:len(grouped_df)]
    return grouped_df


def test_geo_plot_no_data(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data.head(0), title="Test Geographic Plot - No Data"
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_no_coordinates(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    # Remove coordinate columns
    df_no_coords = individual_geo_data.drop(columns=[SchemaColumns.GPS_LATITUDE, SchemaColumns.GPS_LONGITUDE])
    fig = create_geo_plot(df_no_coords, title="Test Geographic Plot - No Coordinates")
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_individual_data(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(individual_geo_data, title="Test Geographic Plot - Individual")
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