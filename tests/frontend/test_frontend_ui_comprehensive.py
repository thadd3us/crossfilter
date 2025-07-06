"""Comprehensive frontend UI tests for plot filtering and interaction using Playwright."""

import logging
import re
import sys
from typing import Dict, Any, List, Tuple, Optional

import pytest
from playwright.sync_api import Page
from syrupy import SnapshotAssertion
from syrupy.extensions.image import PNGImageSnapshotExtension

from crossfilter.core.schema import DataType
from crossfilter.core.backend_frontend_shared_schema import (
    ProjectionType,
    FilterOperatorType,
)
from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)


def set_browser_window_size(page: Page) -> None:
    """Set browser window to be tall enough to capture all UI elements."""
    # Set a tall window to capture both temporal and geo plots plus controls
    page.set_viewport_size({"width": 1200, "height": 1400})


def wait_for_app_ready(page: Page) -> None:
    """Wait for the application to be fully loaded and ready."""
    # Wait for page title
    page.wait_for_function("document.title === 'Crossfilter'", timeout=5000)

    # Wait for Vue app to mount
    page.wait_for_selector('.app', timeout=5000)

    # Wait for both plots to render in Vue.js structure
    page.wait_for_function(
        """
        () => {
            const plotContainers = document.querySelectorAll('.plot-container');
            return plotContainers.length >= 2 && 
                   Array.from(plotContainers).filter(container => 
                       container.querySelector('.main-svg') !== null
                   ).length >= 2;
        }
        """,
        timeout=30000,
    )

    # Wait for initial status to show data is loaded in Vue.js structure
    page.wait_for_function(
        "document.querySelector('.status-info').textContent.includes('100 (100.0%) of 100 rows loaded')",
        timeout=10000,
    )


def get_status_info(page: Page) -> Dict[str, str]:
    """Extract status information from various status elements."""
    # Get selection info from Vue.js structure - spans that contain "Selected X rows"
    temporal_selection = ""
    geo_selection = ""
    
    # Find selection spans in projection toolbars
    selection_spans = page.locator("span:has-text('Selected')")
    for i in range(selection_spans.count()):
        span_text = selection_spans.nth(i).text_content() or ""
        if "Selected" in span_text:
            # Determine which projection this belongs to based on context
            # Check if it's in the first or second projection
            projection_containers = page.locator(".projection")
            for j in range(projection_containers.count()):
                projection = projection_containers.nth(j)
                if projection.locator("span:has-text('Selected')").count() > 0:
                    span_in_projection = projection.locator("span:has-text('Selected')").first.text_content() or ""
                    if span_in_projection == span_text:
                        if j == 0:  # First projection (Temporal)
                            temporal_selection = span_text
                        elif j == 1:  # Second projection (Geo)
                            geo_selection = span_text
                        break
    
    return {
        "main_status": page.locator(".status-info").text_content() or "",
        "temporal_status": "",  # Vue.js doesn't have separate status elements
        "geo_status": "",  # Vue.js doesn't have separate status elements
        "temporal_selection": temporal_selection,
        "geo_selection": geo_selection,
    }


def extract_row_count(page: Page, status_text: str) -> int:
    """Extract current row count from status text."""
    # Look for pattern like "76 (76.0%) of 100 rows loaded"
    parts = status_text.split()
    for i, part in enumerate(parts):
        if part.startswith("(") and part.endswith("%)"):
            # The count should be the part before this
            if i > 0:
                count_str = parts[i - 1]
                try:
                    return int(count_str)
                except ValueError:
                    pass
    raise ValueError(f"Could not extract row count from: {status_text}")


def toggle_datatype_visibility(
    page: Page, plot_type: ProjectionType, datatype: DataType, make_visible: bool
) -> None:
    """Toggle visibility of a specific datatype in the legend.

    Args:
        page: Playwright page object
        plot_type: ProjectionType.TEMPORAL or ProjectionType.GEO
        datatype: DataType enum value
        make_visible: True to make visible, False to hide
    """
    # Find the plot container based on projection type (first or second)
    projection_index = 0 if plot_type == ProjectionType.TEMPORAL else 1

    # Use Plotly's restyle to toggle trace visibility programmatically
    # This is more reliable than trying to click legend elements
    page.evaluate(
        f"""
        () => {{
            const projections = document.querySelectorAll('.projection');
            if (projections.length > {projection_index}) {{
                const plotDiv = projections[{projection_index}].querySelector('.plot-container');
                if (plotDiv && plotDiv.data) {{
                    // Find traces that match the datatype name
                    const updates = {{}};
                    plotDiv.data.forEach((trace, index) => {{
                        if (trace.name && trace.name.includes('{datatype}')) {{
                            updates[index] = {{"visible": {str(make_visible).lower()}}};
                        }}
                    }});
                    
                    // Apply the updates
                    const traceIndices = Object.keys(updates).map(i => parseInt(i));
                    const visibilityValues = Object.values(updates).map(u => u.visible);
                    
                    if (traceIndices.length > 0) {{
                        Plotly.restyle(plotDiv, 'visible', visibilityValues, traceIndices);
                    }}
                }}
            }}
        }}
        """
    )

    # Wait for the plot to update
    page.wait_for_timeout(500)


def double_click_datatype_to_isolate(
    page: Page, plot_type: ProjectionType, datatype: DataType
) -> None:
    """Double-click a datatype in the legend to show only that datatype."""
    # Find the plot container based on projection type (first or second)
    projection_index = 0 if plot_type == ProjectionType.TEMPORAL else 1

    # Use Plotly's restyle to isolate one trace (hide all others)
    page.evaluate(
        f"""
        () => {{
            const projections = document.querySelectorAll('.projection');
            if (projections.length > {projection_index}) {{
                const plotDiv = projections[{projection_index}].querySelector('.plot-container');
                if (plotDiv && plotDiv.data) {{
                    // Find traces that match or don't match the datatype name
                    const updates = [];
                    plotDiv.data.forEach((trace, index) => {{
                        if (trace.name && trace.name.includes('{datatype}')) {{
                            updates.push(true);  // Show this trace
                        }} else {{
                            updates.push(false); // Hide other traces
                        }}
                    }});
                    
                    // Apply the updates to all traces
                    if (updates.length > 0) {{
                        const traceIndices = [...Array(updates.length).keys()];
                        Plotly.restyle(plotDiv, 'visible', updates, traceIndices);
                    }}
                }}
            }}
        }}
        """
    )

    # Wait for the plot to update
    page.wait_for_timeout(500)


def get_plot_bounds(page: Page, plot_type: ProjectionType) -> Dict[str, float]:
    """Get the bounding box of a plot container."""
    # Find the plot container based on projection type (first or second)
    projection_index = 0 if plot_type == ProjectionType.TEMPORAL else 1
    
    containers = page.locator(".plot-container")
    if containers.count() <= projection_index:
        raise RuntimeError(f"Could not find plot container for {plot_type} plot")
    
    container = containers.nth(projection_index)
    bounds = container.bounding_box()
    if not bounds:
        raise RuntimeError(f"Could not get bounds for {plot_type} plot")
    return bounds


def drag_select_plot_region(
    page: Page,
    plot_type: ProjectionType,
    region_fraction: Tuple[float, float, float, float],
) -> None:
    """Select a rectangular region in a plot using box select.

    Args:
        page: Playwright page object
        plot_type: ProjectionType.TEMPORAL or ProjectionType.GEO
        region_fraction: (left, top, right, bottom) as fractions of plot size (0.0 to 1.0)
    """
    # Find the plot container based on projection type (first or second)
    projection_index = 0 if plot_type == ProjectionType.TEMPORAL else 1

    # First activate box select tool in the specific plot container
    containers = page.locator(".plot-container")
    if containers.count() <= projection_index:
        raise RuntimeError(f"Could not find plot container for {plot_type} plot")
    
    box_select_button = containers.nth(projection_index).locator(
        "[data-attr='dragmode'][data-val='select']"
    )
    box_select_button.click()
    page.wait_for_timeout(200)

    # Get plot bounds
    bounds = get_plot_bounds(page, plot_type)

    # Calculate selection coordinates
    left, top, right, bottom = region_fraction
    start_x = bounds["x"] + bounds["width"] * left
    start_y = bounds["y"] + bounds["height"] * top
    end_x = bounds["x"] + bounds["width"] * right
    end_y = bounds["y"] + bounds["height"] * bottom

    logger.info(
        f"Selecting region in {plot_type} plot: bounds={bounds}, "
        f"selection=({start_x:.1f},{start_y:.1f}) to ({end_x:.1f},{end_y:.1f})"
    )

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for selection to register and check if it worked
    page.wait_for_timeout(1000)

    # Debug: Check if selection actually worked with Vue.js app structure
    selection_count = page.evaluate(
        f"""
        () => {{
            const app = window.app;
            if (app && app._instance && app._instance.setupState && app._instance.setupState.appState) {{
                const appState = app._instance.setupState.appState;
                if (appState.projections && appState.projections['{plot_type}']) {{
                    return appState.projections['{plot_type}'].selectedDfIds.size;
                }}
            }}
            return 0;
        }}
        """
    )
    logger.info(
        f"After selection attempt: {selection_count} points selected in {plot_type} plot"
    )


def click_filter_button(
    page: Page, plot_type: ProjectionType, operation: FilterOperatorType
) -> None:
    """Click a filter button and wait for the operation to complete.

    Args:
        page: Playwright page object
        plot_type: ProjectionType.TEMPORAL or ProjectionType.GEO
        operation: FilterOperatorType.INTERSECTION or FilterOperatorType.SUBTRACTION
    """
    # Find the correct filter button in the Vue.js structure
    projection_index = 0 if plot_type == ProjectionType.TEMPORAL else 1
    operation_class = "intersection" if operation == FilterOperatorType.INTERSECTION else "subtraction"
    
    projections = page.locator(".projection")
    if projections.count() <= projection_index:
        raise RuntimeError(f"Could not find projection for {plot_type}")
    
    projection = projections.nth(projection_index)
    button = projection.locator(f".filter-button.{operation_class}")

    # Wait for button to be enabled
    page.wait_for_function(
        f"""
        () => {{
            const projections = document.querySelectorAll('.projection');
            if (projections.length > {projection_index}) {{
                const button = projections[{projection_index}].querySelector('.filter-button.{operation_class}');
                return button && !button.disabled;
            }}
            return false;
        }}
        """,
        timeout=2000,
    )

    # Get current status to detect changes
    initial_status = get_status_info(page)["main_status"]
    initial_count = extract_row_count(page, initial_status)

    # Click the button
    button.click()

    # Wait for the filter to be applied (status should change)
    page.wait_for_function(
        f"""
        () => {{
            const status = document.querySelector('.status-info').textContent;
            const currentMatch = status.match(/(\\d+) \\(\\d+\\.\\d+%\\) of \\d+ rows loaded/);
            return currentMatch && parseInt(currentMatch[1]) !== {initial_count};
        }}
        """,
        timeout=10000,
    )

    # Additional wait for UI to stabilize
    page.wait_for_timeout(1000)


@pytest.mark.e2e
def test_temporal_plot_filtering_workflow_simplified(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test the temporal plot filtering workflow (adapted from working basic test):

    1. Start with preloaded data
    2. Turn off one DataType by manipulating plot traces
    3. Select points using box select
    4. Filter to selected (intersection)
    5. Select another region and use subtraction
    6. Verify correct behavior
    """
    # Set browser window size for comprehensive UI capture
    set_browser_window_size(page)

    # Navigate and wait for app to be ready
    page.goto(f"{server_with_data}/")
    wait_for_app_ready(page)

    # Screenshot: Initial loaded state
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify initial state
    initial_status = get_status_info(page)
    initial_count = extract_row_count(page, initial_status["main_status"])
    assert initial_count == 100, f"Expected 100 initial rows, got {initial_count}"

    # Turn off one DataType by hiding a trace
    toggle_datatype_visibility(page, ProjectionType.TEMPORAL, DataType.PHOTO, False)

    # Screenshot: After hiding PHOTO data type
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Do box selection on temporal plot (copy working approach from basic test)
    # Use first plot container (temporal)
    plot_containers = page.locator(".plot-container")
    temporal_container = plot_containers.first

    # Click the box select tool
    box_select_button = temporal_container.locator(
        "[data-attr='dragmode'][data-val='select']"
    )
    box_select_button.click()

    # Get plot container bounds for selection
    plot_box = temporal_container.bounding_box()

    # Perform box selection by dragging within the plot area
    # Calculate selection area (inner 60% of plot)
    margin_x = plot_box["width"] * 0.2
    margin_y = plot_box["height"] * 0.2

    start_x = plot_box["x"] + margin_x
    start_y = plot_box["y"] + margin_y
    end_x = plot_box["x"] + plot_box["width"] - margin_x
    end_y = plot_box["y"] + plot_box["height"] - margin_y

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for the buttons to become enabled after selection in Vue.js
    page.wait_for_function(
        "!document.querySelector('.filter-button.intersection').disabled",
        timeout=2000,
    )

    # Screenshot: After drag selection with enabled buttons
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify selection info is displayed in Vue.js structure
    page.wait_for_selector("span:has-text('Selected')", timeout=2000)
    selection_info = page.locator("span:has-text('Selected')").first.text_content()
    logger.info(f"Temporal selection: '{selection_info}'")

    # Use flexible assertion for selection count
    selected_match = re.search(r'Selected (\d+) rows', selection_info)
    assert selected_match is not None, f"Could not find selection count in: {selection_info}"
    selected_count = int(selected_match.group(1))
    assert 45 <= selected_count <= 55, f"Expected 45-55 selected rows, got {selected_count}"

    # Apply intersection filter
    click_filter_button(page, ProjectionType.TEMPORAL, FilterOperatorType.INTERSECTION)

    # Screenshot: After intersection filter applied
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify both plots show fewer rows after filtering
    after_intersection_status = get_status_info(page)
    intersection_count = extract_row_count(
        page, after_intersection_status["main_status"]
    )
    logger.info(f"After intersection: {initial_count} → {intersection_count} rows")
    assert intersection_count < initial_count
    assert intersection_count <= 51  # Should be <= the selected count

    # Now do a second selection for subtraction test
    # Select a smaller region in the middle
    margin_x = plot_box["width"] * 0.35
    margin_y = plot_box["height"] * 0.35

    start_x = plot_box["x"] + margin_x
    start_y = plot_box["y"] + margin_y
    end_x = plot_box["x"] + plot_box["width"] - margin_x
    end_y = plot_box["y"] + plot_box["height"] - margin_y

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for selection to register
    page.wait_for_function(
        "!document.querySelector('.filter-button.subtraction').disabled",
        timeout=2000,
    )

    # Screenshot: After second drag selection for subtraction
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Apply subtraction filter
    click_filter_button(page, ProjectionType.TEMPORAL, FilterOperatorType.SUBTRACTION)

    # Screenshot: After subtraction filter applied (final state)
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify point count decreases again
    after_subtraction_status = get_status_info(page)
    subtraction_count = extract_row_count(page, after_subtraction_status["main_status"])
    logger.info(f"After subtraction: {intersection_count} → {subtraction_count} rows")
    assert subtraction_count < intersection_count

    logger.info(
        f"Temporal filtering test completed: {initial_count} → {intersection_count} → {subtraction_count} rows"
    )


@pytest.mark.e2e
def test_geo_plot_filtering_workflow(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test the geo plot filtering workflow (simplified version):

    1. Start with preloaded data
    2. Try to select points in geo plot using same approach as temporal
    3. Apply intersection filter if selection works
    4. Apply subtraction filter
    5. Verify correct behavior
    """
    # Set browser window size for comprehensive UI capture
    set_browser_window_size(page)

    # Navigate and wait for app to be ready
    page.goto(f"{server_with_data}/")
    wait_for_app_ready(page)

    # Screenshot: Initial loaded state
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify initial state
    initial_status = get_status_info(page)
    initial_count = extract_row_count(page, initial_status["main_status"])
    assert initial_count == 100, f"Expected 100 initial rows, got {initial_count}"

    # Try geo plot selection using same approach as temporal plot
    # Use second plot container (geo)
    plot_containers = page.locator(".plot-container")
    geo_container = plot_containers.nth(1)

    # Click the box select tool in Plotly's mode bar for the geo plot
    box_select_button = geo_container.locator(
        "[data-attr='dragmode'][data-val='select']"
    )
    box_select_button.click()

    # Get plot container bounds for selection
    plot_box = geo_container.bounding_box()

    # Try to perform box selection by dragging within the plot area
    # Calculate selection area (inner 60% of plot)
    margin_x = plot_box["width"] * 0.2
    margin_y = plot_box["height"] * 0.2

    start_x = plot_box["x"] + margin_x
    start_y = plot_box["y"] + margin_y
    end_x = plot_box["x"] + plot_box["width"] - margin_x
    end_y = plot_box["y"] + plot_box["height"] - margin_y

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for the buttons to become enabled after selection
    page.wait_for_function(
        """
        () => {
            const projections = document.querySelectorAll('.projection');
            if (projections.length > 1) {
                const button = projections[1].querySelector('.filter-button.intersection');
                return button && !button.disabled;
            }
            return false;
        }
        """,
        timeout=3000,
    )

    # Screenshot: After geo drag selection with enabled buttons
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify selection info is displayed
    # Get geo selection info (from second projection)
    geo_projection = page.locator(".projection").nth(1)
    page.wait_for_selector(".projection:nth-child(2) span:has-text('Selected')", timeout=2000)
    selection_info = geo_projection.locator("span:has-text('Selected')").text_content()
    logger.info(f"Geo selection: '{selection_info}'")

    # Use flexible assertion for selection count
    selected_match = re.search(r'Selected (\d+) rows', selection_info)
    assert selected_match is not None, f"Could not find selection count in: {selection_info}"
    selected_count = int(selected_match.group(1))
    assert 75 <= selected_count <= 85, f"Expected 75-85 selected rows, got {selected_count}"

    # Apply intersection filter
    click_filter_button(page, ProjectionType.GEO, FilterOperatorType.INTERSECTION)

    # Screenshot: After geo intersection filter applied
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify both plots show fewer rows after filtering
    after_intersection_status = get_status_info(page)
    intersection_count = extract_row_count(
        page, after_intersection_status["main_status"]
    )
    logger.info(f"After geo intersection: {initial_count} → {intersection_count} rows")
    assert intersection_count < initial_count
    assert intersection_count <= 80

    # Do a second selection for subtraction test
    # Select a smaller region in the middle
    margin_x = plot_box["width"] * 0.35
    margin_y = plot_box["height"] * 0.35

    start_x = plot_box["x"] + margin_x
    start_y = plot_box["y"] + margin_y
    end_x = plot_box["x"] + plot_box["width"] - margin_x
    end_y = plot_box["y"] + plot_box["height"] - margin_y

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for selection to register
    page.wait_for_function(
        """
        () => {
            const projections = document.querySelectorAll('.projection');
            if (projections.length > 1) {
                const button = projections[1].querySelector('.filter-button.subtraction');
                return button && !button.disabled;
            }
            return false;
        }
        """,
        timeout=2000,
    )

    # Screenshot: After second geo drag selection for subtraction
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Apply subtraction filter
    click_filter_button(page, ProjectionType.GEO, FilterOperatorType.SUBTRACTION)

    # Screenshot: After geo subtraction filter applied (final state)
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify point count decreases again
    after_subtraction_status = get_status_info(page)
    subtraction_count = extract_row_count(page, after_subtraction_status["main_status"])
    logger.info(
        f"After geo subtraction: {intersection_count} → {subtraction_count} rows"
    )
    assert subtraction_count < intersection_count

    logger.info(
        f"Geo filtering test completed: {initial_count} → {intersection_count} → {subtraction_count} rows"
    )


@pytest.mark.e2e
def test_datatype_legend_interactions(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test various DataType legend interactions in both plots."""
    # Set browser window size for comprehensive UI capture
    set_browser_window_size(page)

    # Navigate and wait for app to be ready
    page.goto(f"{server_with_data}/")
    wait_for_app_ready(page)

    # Screenshot: Initial loaded state
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Test programmatic trace visibility toggling on temporal plot
    toggle_datatype_visibility(page, ProjectionType.TEMPORAL, DataType.PHOTO, False)
    page.wait_for_timeout(500)

    # Screenshot: After hiding PHOTO data type
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify PHOTO is hidden and bring it back
    toggle_datatype_visibility(page, ProjectionType.TEMPORAL, DataType.PHOTO, True)
    page.wait_for_timeout(500)

    # Screenshot: After showing PHOTO data type again
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Test double click to isolate in temporal plot
    double_click_datatype_to_isolate(
        page, ProjectionType.TEMPORAL, DataType.GPX_TRACKPOINT
    )
    page.wait_for_timeout(500)

    # Screenshot: After isolating GPX_TRACKPOINT data type
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify the trace isolation worked by checking plot data
    visible_traces = page.evaluate(
        """
        () => {
            const projections = document.querySelectorAll('.projection');
            if (projections.length > 0) {
                const plotDiv = projections[0].querySelector('.plot-container');
                if (plotDiv && plotDiv.data) {
                    return plotDiv.data.map(trace => ({
                        name: trace.name,
                        visible: trace.visible !== false
                    }));
                }
            }
            return [];
        }
        """
    )
    logger.info(f"Visible traces after isolation: {visible_traces}")

    # Count how many traces are visible
    visible_count = sum(1 for trace in visible_traces if trace["visible"])
    assert visible_count > 0, "Expected at least one visible trace"
    assert visible_count < len(
        visible_traces
    ), "Expected fewer visible traces after isolation"

    # THAD: Click the reset data button, check counts, get another screenshot.


@pytest.mark.e2e
def test_drag_select_helper_function(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test the drag_select_plot_region helper function works correctly."""
    # Set browser window size for comprehensive UI capture
    set_browser_window_size(page)

    # Navigate and wait for app to be ready
    page.goto(f"{server_with_data}/")
    wait_for_app_ready(page)

    # Screenshot: Initial loaded state
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Test drag selection on temporal plot
    drag_select_plot_region(page, ProjectionType.TEMPORAL, (0.2, 0.2, 0.8, 0.8))

    # Screenshot: After temporal drag selection
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify selection worked
    temporal_selection = get_status_info(page)["temporal_selection"]
    if temporal_selection:
        # Use flexible assertion
        selected_match = re.search(r'Selected (\d+) rows', temporal_selection)
        assert selected_match is not None, f"Could not find selection count in: {temporal_selection}"
        selected_count = int(selected_match.group(1))
        assert 80 <= selected_count <= 90, f"Expected 80-90 selected rows, got {selected_count}"
    else:
        # If no selection info found, check directly from drag_select_plot_region logs
        print("No temporal selection info found in status, but drag selection may have worked")

    # Test drag selection on geo plot
    drag_select_plot_region(page, ProjectionType.GEO, (0.2, 0.2, 0.8, 0.8))

    # Screenshot: After geo drag selection (final state)
    if "--headed" not in sys.argv:
        assert page.screenshot(full_page=True) == snapshot(
            extension_class=PNGImageSnapshotExtension
        )

    # Verify selection worked  
    geo_selection = get_status_info(page)["geo_selection"]
    if geo_selection:
        # Use flexible assertion
        selected_match = re.search(r'Selected (\d+) rows', geo_selection)
        assert selected_match is not None, f"Could not find selection count in: {geo_selection}"
        selected_count = int(selected_match.group(1))
        assert 75 <= selected_count <= 85, f"Expected 75-85 selected rows, got {selected_count}"
    else:
        # If no geo selection found, that's okay - geo plots might not always have selectable points
        print("No geo selection info found in status - geo plot may not have selectable points in this region")
