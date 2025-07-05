"""Tests for geo plot module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension

from crossfilter.core.schema import SchemaColumns, load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.geo_plot import (
    _calculate_geographic_bounds,
    _calculate_geographic_center,
    _calculate_map_view,
    _calculate_zoom_level,
    create_geo_plot,
)


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


class TestGeographicCenter:
    """Test geographic center calculations."""

    def test_simple_center(self) -> None:
        """Test center calculation for simple cases."""
        # Test points in same hemisphere
        lats = pd.Series([40.0, 42.0])
        lons = pd.Series([-74.0, -72.0])
        
        center_lat, center_lon = _calculate_geographic_center(lats, lons)
        
        # Should be approximately the midpoint
        assert abs(center_lat - 41.0) < 0.1
        assert abs(center_lon - (-73.0)) < 0.1

    def test_date_line_crossing(self) -> None:
        """Test center calculation when crossing the International Date Line."""
        # Points on either side of date line
        lats = pd.Series([40.0, 40.0])
        lons = pd.Series([-179.0, 179.0])
        
        center_lat, center_lon = _calculate_geographic_center(lats, lons)
        
        # Center should be at latitude 40, longitude 180 (or -180)
        assert abs(center_lat - 40.0) < 0.01
        # Should be near the date line, not at longitude 0
        assert abs(abs(center_lon) - 180.0) < 1.0

    def test_polar_regions(self) -> None:
        """Test center calculation near poles."""
        # Points near north pole
        lats = pd.Series([85.0, 87.0])
        lons = pd.Series([0.0, 180.0])
        
        center_lat, center_lon = _calculate_geographic_center(lats, lons)
        
        # Should be high latitude
        assert center_lat > 85.0

    def test_single_point(self) -> None:
        """Test center calculation with single point."""
        lats = pd.Series([40.7128])
        lons = pd.Series([-74.0060])
        
        center_lat, center_lon = _calculate_geographic_center(lats, lons)
        
        # Should be exactly the point
        assert abs(center_lat - 40.7128) < 0.0001
        assert abs(center_lon - (-74.0060)) < 0.0001


class TestGeographicBounds:
    """Test geographic bounds calculations."""

    def test_simple_bounds(self) -> None:
        """Test bounds calculation for simple cases."""
        lats = pd.Series([40.0, 42.0])
        lons = pd.Series([-74.0, -72.0])
        
        lat_min, lat_max, lon_min, lon_max = _calculate_geographic_bounds(lats, lons)
        
        assert lat_min == 40.0
        assert lat_max == 42.0
        assert lon_min == -74.0
        assert lon_max == -72.0

    def test_date_line_crossing(self) -> None:
        """Test bounds calculation when crossing the International Date Line."""
        lats = pd.Series([40.0, 40.0, 40.0])
        lons = pd.Series([-179.0, 179.0, 178.0])
        
        lat_min, lat_max, lon_min, lon_max = _calculate_geographic_bounds(lats, lons)
        
        # Should handle date line crossing properly
        assert lat_min == 40.0
        assert lat_max == 40.0
        # With simplified bounds, we get the actual min/max which spans wide
        assert lon_min == -179.0
        assert lon_max == 179.0

    def test_no_date_line_crossing(self) -> None:
        """Test bounds when points span wide but don't cross date line."""
        lats = pd.Series([0.0, 0.0])
        lons = pd.Series([-120.0, 120.0])
        
        lat_min, lat_max, lon_min, lon_max = _calculate_geographic_bounds(lats, lons)
        
        # Should be normal bounds
        assert lat_min == 0.0
        assert lat_max == 0.0
        assert lon_min == -120.0
        assert lon_max == 120.0


class TestZoomLevel:
    """Test zoom level calculations."""

    def test_global_zoom(self) -> None:
        """Test zoom level for global data."""
        zoom = _calculate_zoom_level(180.0, 360.0, 0.0)
        assert zoom <= 2  # Should be very zoomed out

    def test_city_zoom(self) -> None:
        """Test zoom level for city-scale data."""
        zoom = _calculate_zoom_level(0.5, 0.5, 40.0)
        assert zoom >= 8  # Should be zoomed in

    def test_mercator_adjustment(self) -> None:
        """Test that longitude spans are adjusted for Mercator projection."""
        # Same degree spans but different latitudes
        zoom_equator = _calculate_zoom_level(1.0, 1.0, 0.0)
        zoom_high_lat = _calculate_zoom_level(1.0, 1.0, 60.0)
        
        # Higher latitude should have higher zoom (more zoomed in)
        # because longitude degrees are "narrower" there
        assert zoom_high_lat >= zoom_equator

    def test_single_point_zoom(self) -> None:
        """Test zoom level for very small spans."""
        zoom = _calculate_zoom_level(0.001, 0.001, 40.0)
        assert zoom >= 12  # Should be very zoomed in


class TestMapView:
    """Test integrated map view calculations."""

    def test_normal_case(self) -> None:
        """Test map view calculation for normal case."""
        lats = pd.Series([40.0, 42.0])
        lons = pd.Series([-74.0, -72.0])
        
        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)
        
        # Should be reasonable values
        assert 40.0 <= center_lat <= 42.0
        assert -74.0 <= center_lon <= -72.0
        assert 1 <= zoom <= 14

    def test_date_line_case(self) -> None:
        """Test map view calculation when crossing date line."""
        lats = pd.Series([40.0, 40.0])
        lons = pd.Series([-179.0, 179.0])
        
        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)
        
        # Should handle date line crossing
        assert abs(center_lat - 40.0) < 0.1
        assert abs(abs(center_lon) - 180.0) < 10.0  # Near date line
        assert 1 <= zoom <= 14

    def test_single_point_case(self) -> None:
        """Test map view calculation for single point."""
        lats = pd.Series([40.7128])
        lons = pd.Series([-74.0060])
        
        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)
        
        # Should be exactly the point with high zoom
        assert abs(center_lat - 40.7128) < 0.0001
        assert abs(center_lon - (-74.0060)) < 0.0001
        assert zoom >= 12  # Should be very zoomed in


@pytest.fixture
def date_line_crossing_data() -> pd.DataFrame:
    """Create test data that crosses the International Date Line."""
    import datetime
    
    data = []
    # Points crossing the date line around Pacific islands
    locations = [
        ("Western Pacific", 15.0, 179.5),  # Just west of date line
        ("Eastern Pacific", 15.0, -179.5),  # Just east of date line
        ("Kiribati", 1.0, 180.0),  # Exactly on date line
        ("Samoa", -14.0, -171.0),  # Further east
        ("Tonga", -21.0, -175.0),  # Further east
    ]
    
    for i, (name, lat, lon) in enumerate(locations):
        data.append({
            SchemaColumns.UUID_STRING: f"dateline_uuid_{i}",
            SchemaColumns.DATA_TYPE: "PHOTO",
            SchemaColumns.NAME: name,
            SchemaColumns.CAPTION: f"Photo at {name}",
            SchemaColumns.SOURCE_FILE: f"dateline_file_{i}.jpg",
            SchemaColumns.TIMESTAMP_MAYBE_TIMEZONE_AWARE: "2024-01-15T12:00:00",
            SchemaColumns.TIMESTAMP_UTC: datetime.datetime(
                2024, 1, 15, 12, i, 0, tzinfo=datetime.timezone.utc
            ),
            SchemaColumns.GPS_LATITUDE: lat,
            SchemaColumns.GPS_LONGITUDE: lon,
            SchemaColumns.RATING_0_TO_5: 4,
            SchemaColumns.SIZE_IN_BYTES: 1024000,
            SchemaColumns.COUNT: 1,
        })
    
    df = pd.DataFrame(data)
    df.index.name = SchemaColumns.DF_ID
    return df


def test_geo_plot_date_line_crossing(
    snapshot: SnapshotAssertion, date_line_crossing_data: pd.DataFrame
) -> None:
    """Test that plot handles date line crossing correctly."""
    fig = create_geo_plot(
        date_line_crossing_data,
        title="Test Geographic Plot - Date Line Crossing",
    )
    
    # Check that the center is reasonable (near Pacific, not at longitude 0)
    center_lon = fig.layout.map.center.lon
    assert abs(abs(center_lon) - 180.0) < 20.0  # Should be near date line
    
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
