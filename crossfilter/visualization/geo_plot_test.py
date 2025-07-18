"""Tests for geo plot module."""

from pathlib import Path

import pandas as pd
import pytest
from syrupy import SnapshotAssertion

from crossfilter.core.geo_projection_state import GeoProjectionState
from crossfilter.core.schema import SchemaColumns, load_jsonl_to_dataframe
from crossfilter.core.session_state import SessionState
from crossfilter.visualization.geo_plot import (
    _calculate_geographic_center_and_radius,
    _calculate_map_view,
    _calculate_zoom_level_from_radius,
    create_geo_plot,
)
from tests.util.syrupy_html_snapshot import HTMLSnapshotExtension


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
        individual_geo_data.head(0),
        title="Test Geographic Plot - No Data",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_individual_data(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data,
        title="Test Geographic Plot - Individual",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_individual_data_grouped(
    snapshot: SnapshotAssertion, individual_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        individual_geo_data,
        title="Test Geographic Plot - Individual, Grouped By Type",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_with_count_data(
    snapshot: SnapshotAssertion, grouped_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        grouped_geo_data,
        title="Test Geographic Plot - With Count Data",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_with_count_data_grouped(
    snapshot: SnapshotAssertion, grouped_geo_data: pd.DataFrame
) -> None:
    fig = create_geo_plot(
        grouped_geo_data,
        title="Test Geographic Plot - With Count Data, Grouped By Type",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


def test_geo_plot_nyc_area_auto_fit(
    snapshot: SnapshotAssertion, nyc_area_data: pd.DataFrame
) -> None:
    """Test that plot auto-fits to NYC area data, demonstrating zoom behavior."""
    fig = create_geo_plot(
        nyc_area_data,
        title="Test Geographic Plot - NYC Area Auto-Fit",
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )
    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)


class TestGeographicCenterAndRadius:
    """Test geographic center and radius calculations using GeoPandas."""

    def test_simple_center_and_radius(self) -> None:
        """Test center and radius calculation for simple cases."""
        # Test points in same hemisphere
        lats = pd.Series([40.0, 42.0])
        lons = pd.Series([-74.0, -72.0])

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )

        # Should be approximately the midpoint
        assert abs(center_lat - 41.0) < 0.1
        assert abs(center_lon - (-73.0)) < 0.1
        # Radius should be reasonable for points ~200km apart
        assert 100_000 < radius_meters < 300_000

    def test_date_line_crossing(self) -> None:
        """Test center calculation when crossing the International Date Line."""
        # Points on either side of date line
        lats = pd.Series([40.0, 40.0])
        lons = pd.Series([-179.0, 179.0])

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )

        # Center should be at latitude 40, longitude near ±180
        assert abs(center_lat - 40.0) < 0.1
        # Should be near the date line, not at longitude 0
        assert abs(abs(center_lon) - 180.0) < 10.0
        # Radius should be small since points are close across the date line
        assert radius_meters < 500_000  # Less than 500km

    def test_single_point(self) -> None:
        """Test center calculation with single point."""
        lats = pd.Series([40.7128])
        lons = pd.Series([-74.0060])

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )

        # Should be exactly the point
        assert abs(center_lat - 40.7128) < 0.0001
        assert abs(center_lon - (-74.0060)) < 0.0001
        # Radius should be zero for single point
        assert radius_meters == 0.0


class TestZoomLevelFromRadius:
    """Test zoom level calculations based on radius and Mapbox formula."""

    def test_pole_to_pole_zoom_zero(self) -> None:
        """A point on the equator and a point at the north pole results in a zoom level of 0."""
        # Distance from equator to north pole is ~5,000 km (quarter of Earth's circumference)
        radius_meters = 5_000_000  # 5,000 km
        center_lat = 45.0  # Middle latitude

        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)
        assert zoom <= 2  # Should be very zoomed out

    def test_equator_to_south_pole_zoom_zero(self) -> None:
        """A point on the equator and a point at the south pole results in a zoom level of 0."""
        # Distance from equator to south pole is ~5,000 km
        radius_meters = 5_000_000  # 5,000 km
        center_lat = -45.0  # Middle latitude in southern hemisphere

        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)
        assert zoom <= 2  # Should be very zoomed out

    def test_date_line_crossing_high_zoom(self) -> None:
        """Two nearby points on the equator spanning +179/-179 degrees have a high zoom factor."""
        lats = pd.Series([0.0, 0.0])  # Equator
        lons = pd.Series([179.0, -179.0])  # Just 2 degrees apart across date line

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )
        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)

        # Should have high zoom since points are close
        assert zoom >= 8

    def test_equator_points_same_zoom(self) -> None:
        """Two nearby points at +1/-1 degrees should have same zoom as +179/-179."""
        # Points at +1/-1 degrees on equator
        lats1 = pd.Series([0.0, 0.0])
        lons1 = pd.Series([1.0, -1.0])

        center_lat1, center_lon1, radius_meters1 = (
            _calculate_geographic_center_and_radius(lats1, lons1)
        )
        zoom1 = _calculate_zoom_level_from_radius(radius_meters1, center_lat1)

        # Points at +179/-179 degrees on equator
        lats2 = pd.Series([0.0, 0.0])
        lons2 = pd.Series([179.0, -179.0])

        center_lat2, center_lon2, radius_meters2 = (
            _calculate_geographic_center_and_radius(lats2, lons2)
        )
        zoom2 = _calculate_zoom_level_from_radius(radius_meters2, center_lat2)

        # Should have similar zoom levels (within 1 level)
        assert abs(zoom1 - zoom2) <= 1

    def test_north_pole_points_high_zoom(self) -> None:
        """Two nearby points near the north pole have a reasonable zoom level."""
        lats = pd.Series([89.9, 89.95])  # Very close to north pole
        lons = pd.Series([0.0, 1.0])  # Small longitude difference

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )
        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)

        # Should have higher zoom than global scale
        assert zoom >= 3

    def test_south_pole_points_high_zoom(self) -> None:
        """Two nearby points near the south pole have a reasonable zoom level."""
        lats = pd.Series([-89.9, -89.95])  # Very close to south pole
        lons = pd.Series([0.0, 1.0])  # Small longitude difference

        center_lat, center_lon, radius_meters = _calculate_geographic_center_and_radius(
            lats, lons
        )
        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)

        # Should have higher zoom than global scale
        assert zoom >= 3

    def test_small_radius_high_zoom(self) -> None:
        """Very small radius should result in high zoom."""
        radius_meters = 100  # 100 meters
        center_lat = 40.0

        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)
        assert zoom >= 15

    def test_large_radius_low_zoom(self) -> None:
        """Very large radius should result in low zoom."""
        radius_meters = 20_000_000  # 20,000 km (half the Earth)
        center_lat = 0.0

        zoom = _calculate_zoom_level_from_radius(radius_meters, center_lat)
        assert zoom <= 2


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
        assert 0 <= zoom <= 20

    def test_date_line_case(self) -> None:
        """Test map view calculation when crossing date line."""
        lats = pd.Series([40.0, 40.0])
        lons = pd.Series([-179.0, 179.0])

        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)

        # Should handle date line crossing
        assert abs(center_lat - 40.0) < 0.1
        assert abs(abs(center_lon) - 180.0) < 10.0  # Near date line
        assert 0 <= zoom <= 20

    def test_single_point_case(self) -> None:
        """Test map view calculation for single point."""
        lats = pd.Series([40.7128])
        lons = pd.Series([-74.0060])

        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)

        # Should be exactly the point with high zoom
        assert abs(center_lat - 40.7128) < 0.0001
        assert abs(center_lon - (-74.0060)) < 0.0001
        assert zoom == 14  # High zoom for single point

    def test_empty_case(self) -> None:
        """Test map view calculation for empty data."""
        lats = pd.Series([], dtype=float)
        lons = pd.Series([], dtype=float)

        center_lat, center_lon, zoom = _calculate_map_view(lats, lons)

        # Should return default values
        assert center_lat == 0.0
        assert center_lon == 0.0
        assert zoom == 1


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
        data.append(
            {
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
            }
        )

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
        geo_projection_state=GeoProjectionState(max_rows=10_000),
    )

    # Check that the center is reasonable (near Pacific, not at longitude 0)
    center_lon = fig.layout.map.center.lon
    assert abs(abs(center_lon) - 180.0) < 20.0  # Should be near date line

    html_content = fig.to_html(include_plotlyjs="cdn", div_id="test-plot-div")
    assert html_content == snapshot(extension_class=HTMLSnapshotExtension)
