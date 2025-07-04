"""Basic frontend UI test for temporal CDF plot using Playwright."""

import logging

import pytest
from playwright.sync_api import Page
from syrupy import SnapshotAssertion
from syrupy.extensions.image import PNGImageSnapshotExtension

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)

# NOTE: BAD Example of an overly verbose LLM.
# class TemporalCDFPNGExtension(PNGImageSnapshotExtension):
#     """Custom PNG snapshot extension for temporal CDF plots."""

#     _file_extension = "png"


# @pytest.fixture(scope="session")
# def browser(playwright: Playwright) -> Browser:
#     """Use this to get the devtools open immediately.
#     return playwright.chromium.launch(headless=False, devtools=True)


@pytest.mark.skip(reason="Very brittle")
@pytest.mark.e2e
def test_temporal_cdf_plot_png_snapshot(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test that the temporal CDF plot loads and displays correctly."""
    server_url = server_with_data

    # Navigate to the main application page
    page.goto(f"{server_url}/")

    # Wait for the page title to be set
    page.wait_for_function("document.title === 'Crossfilter'", timeout=5000)

    # Wait for the app to initialize, detect pre-loaded data, and auto-load the plot
    # The plot should render automatically when data is detected
    page.wait_for_function(
        """
        () => {
            const plotContainer = document.getElementById('plotContainer');
            return plotContainer && plotContainer.querySelector('.main-svg') !== null;
        }
    """,
        timeout=5000,
    )
    # Take a screenshot of the entire page
    screenshot_bytes = page.screenshot(full_page=True)

    # Use syrupy to compare the screenshot
    assert screenshot_bytes == snapshot(extension_class=PNGImageSnapshotExtension)


@pytest.mark.e2e
def test_temporal_cdf_plot_content(page: Page, server_with_data: str) -> None:
    """Test that the main application page contains expected content and loads the plot."""
    server_url = server_with_data

    # Navigate to the main application page
    page.goto(f"{server_url}/")

    # Check the page title
    assert page.title() == "Crossfilter"

    # Check the main heading
    heading = page.locator("h1").text_content()
    assert heading == "Crossfilter"

    # Check the subtitle
    subtitle = page.locator("p").text_content()
    assert subtitle == "Interactive data exploration, filtering, and selection"

    # Wait for the app to initialize, detect pre-loaded data, and auto-load the plot
    # We wait for the plot to render instead of waiting for button state
    page.wait_for_function(
        """
        () => {
            const plotContainer = document.getElementById('plotContainer');
            return plotContainer && plotContainer.querySelector('.main-svg') !== null;
        }
    """,
        timeout=30000,
    )

    # Verify that status shows data is loaded
    status_text = page.locator("#status").text_content()
    assert "Data loaded" in status_text
    assert "100 rows" in status_text

    # Verify the plot container exists and has content
    plot_container = page.locator("#plotContainer")
    assert plot_container.count() > 0
    assert plot_container.locator(".main-svg").count() > 0


@pytest.mark.e2e
def test_filter_to_selected_ui_elements(page: Page, server_with_data: str) -> None:
    """Test the complete Filter to Selected workflow including:

    1. Rendering the plot with selection tools
    2. Using Plotly box select tool
    3. Performing selection (via drag or programmatic simulation)
    4. Enabling the filter button when points are selected
    5. Clicking the filter button to trigger the filter request
    6. Verifying the UI workflow is complete and functional

    Note: This test focuses on UI workflow validation rather than
    full end-to-end filtering due to JavaScript app initialization
    challenges in the test environment.
    """
    server_url = server_with_data

    # Navigate to the main application page
    page.goto(f"{server_url}/")

    # Wait for the app to initialize, detect pre-loaded data, and auto-load the plot
    # Wait for the plot toolbar to be visible (increased timeout for plot rendering)
    page.wait_for_selector(".modebar", timeout=5000)

    # Check that Filter to Selected button is initially disabled (no selection)
    filter_button = page.locator("#filterToSelectedBtn")
    assert filter_button.is_disabled()

    # Verify plot selection info element exists and is initially empty
    plot_selection_info = page.locator("#plotSelectionInfo")
    assert plot_selection_info.count() > 0
    assert plot_selection_info.text_content() == ""

    # Get initial row count to verify data is loaded
    initial_status = page.locator("#status").text_content() or "NO TEXT CONTENT"
    assert "100 after filtering" in initial_status

    # Now perform the full selection workflow:

    # Click the box select tool in Plotly's mode bar
    box_select_button = page.locator("[data-attr='dragmode'][data-val='select']")
    box_select_button.click()

    # 2. Get plot container bounds for selection
    plot_container = page.locator("#plotContainer")
    plot_box = plot_container.bounding_box()

    # 3. Try to perform box selection by dragging within the plot area
    # Calculate selection area (inner 60% of plot)
    margin_x = plot_box["width"] * 0.2
    margin_y = plot_box["height"] * 0.2

    start_x = plot_box["x"] + margin_x
    start_y = plot_box["y"] + margin_y
    end_x = plot_box["x"] + plot_box["width"] - margin_x
    end_y = plot_box["y"] + plot_box["height"] - margin_y

    # page.evaluate(
    #     """
    #     document.querySelector('#plotContainer').addEventListener('mousedown', e => console.log('mousedown', e));
    #     document.querySelector('#plotContainer').addEventListener('mouseup', e => console.log('mouseup', e));
    #     // document.querySelector('#plotContainer').addEventListener('mousemove', e => console.log('mousemove', e));
    #     """
    # )

    # Perform drag selection
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    # Wait for the button to become enabled after selection
    page.wait_for_function(
        "!document.getElementById('filterToSelectedBtn').disabled", timeout=1000
    )

    # Verify selection info is displayed
    selection_info = plot_selection_info.text_content()
    assert "Selected: 36 points" in selection_info

    # Click the filterToSelectedBtn and debug JavaScript execution
    logger.info("🔍 Debugging JavaScript execution before click...")

    # Check JavaScript console for any errors
    js_debug = page.evaluate(
        """
        () => {
            const btn = document.getElementById('filterToSelectedBtn');
            const app = window.app;

            return {
                buttonExists: !!btn,
                buttonDisabled: btn ? btn.disabled : 'no button',
                buttonOnclick: btn ? btn.getAttribute('onclick') : 'no button',
                hasApp: !!app,
                hasFilterFunction: !!(app && app.filterTemporalToSelected),
                selectedCount: app ? app.selectedRowIndices.size : 'no app',
                hasPlotData: !!(app && app.plotData)
            };
        }
    """
    )
    logger.info(f"JavaScript state: {js_debug}")

    filter_button.click()
    logger.info("✓ Button clicked")

    # Wait for the POST request to be sent and response received
    logger.info("⏳ Waiting for filter request and response...")

    try:
        page.wait_for_function(
            """
            () => {
                const status = document.getElementById('status').textContent;
                // Check if filtering has reduced the count (not showing "100 after filtering" anymore)
                return status.includes('after filtering') && !status.includes('100 after filtering');
            }
            """,
            timeout=1000,
        )

        filtered_status = page.locator("#status").text_content() or "NO TEXT CONTENT"
        assert "36 after filtering" in filtered_status
    finally:
        current_status = page.locator("#status").text_content()
        logger.info(f"Current status: {current_status}")

        # Check if the issue is that SSE is not implemented
        has_event_source = page.evaluate(
            "typeof EventSource !== 'undefined' && !!window.EventSource"
        )
        sse_connections = page.evaluate(
            """
            () => {
                // Check if there are any EventSource connections
                return document.querySelector('script[src*="eventsource"]') !== null ||
                       window.eventSource !== undefined;
            }
        """
        )

        logger.info(f"EventSource available: {has_event_source}")
        logger.info(f"SSE connections detected: {sse_connections}")

        # Check if SSE is working correctly
        sse_status = page.evaluate(
            """
            () => {
                return {
                    eventSourceExists: !!window.app.eventSource,
                    readyState: window.app.eventSource ? window.app.eventSource.readyState : 'no eventSource',
                    filterVersion: window.app.filterVersion
                };
            }
        """
        )

        logger.info(f"SSE Status: {sse_status}")
