"""Basic frontend UI test for temporal CDF plot using Playwright."""

import logging
import sys

import pytest
from playwright.sync_api import Page
from syrupy import SnapshotAssertion
from syrupy.extensions.image import PNGImageSnapshotExtension

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)


# @pytest.fixture(scope="session")
# def browser(playwright: Playwright) -> Browser:
#     """Use this to get the devtools open immediately.
#     return playwright.chromium.launch(headless=False, devtools=True)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Only runs this on macOS -- it is brittle across multiple platforms.",
)
@pytest.mark.e2e
def test_temporal_cdf_plot_png_snapshot(
    page: Page, server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test that the temporal CDF plot loads and displays correctly."""

    # Skip if running in headed mode (visible browser)
    # Check if --headed flag is present in command line arguments
    if "--headed" in sys.argv:
        pytest.skip(
            "Skipping in headed mode -- headed browsers can interfere with screenshot consistency."
        )

    server_url = server_with_data

    # Navigate to the main application page
    page.goto(f"{server_url}/")

    # Wait for the page title to be set
    page.wait_for_function("document.title === 'Crossfilter'", timeout=5000)

    # Wait for Vue app to mount and initialize
    page.wait_for_selector('.app', timeout=5000)

    # Wait for the app to initialize, detect pre-loaded data, and auto-load the plot
    # The plot should render automatically when data is detected in Vue.js structure
    page.wait_for_function(
        """
        () => {
            const plotContainers = document.querySelectorAll('.plot-container');
            return plotContainers.length > 0 &&
                   Array.from(plotContainers).some(container =>
                       container.querySelector('.main-svg') !== null
                   );
        }
    """,
        timeout=10000,
    )
    page.wait_for_timeout(1000)

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

    # Wait for Vue app to mount
    page.wait_for_selector('.app', timeout=5000)

    # Check that the Vue app structure is present
    app_container = page.locator('.app')
    assert app_container.count() > 0

    # Check that status bar exists
    status_bar = page.locator('.top-status-bar')
    assert status_bar.count() > 0

    # Wait for the app to initialize, detect pre-loaded data, and auto-load the plot
    # We wait for the plot to render instead of waiting for button state
    page.wait_for_function(
        """
        () => {
            const plotContainers = document.querySelectorAll('.plot-container');
            return plotContainers.length > 0 &&
                   Array.from(plotContainers).some(container =>
                       container.querySelector('.main-svg') !== null
                   );
        }
    """,
        timeout=30000,
    )

    # Verify that status shows data is loaded with new Vue.js format
    status_text = page.locator(".status-info").text_content()
    assert "100 (100.0%) of 100 rows loaded" in status_text
    assert "cols" in status_text
    assert "MB" in status_text

    # Verify the plot containers exist and have content
    plot_containers = page.locator(".plot-container")
    assert plot_containers.count() > 0
    # At least one plot container should have a plotly plot
    page.wait_for_function(
        "document.querySelectorAll('.plot-container .main-svg').length > 0",
        timeout=5000
    )


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

    # Check that filter buttons are initially disabled (no selection) in Vue.js structure
    intersection_button = page.locator(".filter-button.intersection").first
    subtraction_button = page.locator(".filter-button.subtraction").first
    assert intersection_button.is_disabled()
    assert subtraction_button.is_disabled()

    # In Vue.js, selection info is shown inline when there's a selection
    # Initially there should be no selection text visible
    selection_spans = page.locator("span:has-text('Selected')")
    assert selection_spans.count() == 0

    # Get initial row count to verify data is loaded from Vue.js status bar
    initial_status = page.locator(".status-info").text_content() or "NO TEXT CONTENT"
    assert "100 (100.0%) of 100 rows loaded" in initial_status

    # Now perform the full selection workflow:

    # Click the box select tool in Plotly's mode bar for the first projection (temporal)
    # Target the box select button within the first plot container
    box_select_button = page.locator(
        ".plot-container [data-attr='dragmode'][data-val='select']"
    ).first
    box_select_button.click()

    # 2. Get plot container bounds for selection (first plot container)
    plot_container = page.locator(".plot-container").first
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

    # Wait for the buttons to become enabled after selection in Vue.js
    page.wait_for_function(
        "!document.querySelector('.filter-button.intersection').disabled",
        timeout=2000,
    )

    # Verify selection info is displayed in Vue.js structure
    page.wait_for_selector("span:has-text('Selected')", timeout=2000)
    selection_info = page.locator("span:has-text('Selected')").first.text_content()
    # Check that we have a reasonable selection (between 80-90 rows)
    import re
    selected_match = re.search(r'Selected (\d+) rows', selection_info)
    assert selected_match is not None, f"Could not find selection count in: {selection_info}"
    selected_count = int(selected_match.group(1))
    assert 80 <= selected_count <= 90, f"Expected 80-90 selected rows, got {selected_count}"

    # Click the intersection button and debug JavaScript execution
    logger.info("🔍 Debugging JavaScript execution before click...")

    # Check JavaScript console for any errors with Vue.js app
    js_debug = page.evaluate(
        """
        () => {
            const btn = document.querySelector('.filter-button.intersection');
            const app = window.app;

            return {
                buttonExists: !!btn,
                buttonDisabled: btn ? btn.disabled : 'no button',
                hasApp: !!app,
                hasVueApp: !!app && !!app._instance,
                appStateExists: !!app && !!app._instance && !!app._instance.appState,
                hasProjections: !!app && !!app._instance && !!app._instance.appState && !!app._instance.appState.projections,
                temporalProjectionExists: !!app && !!app._instance && !!app._instance.appState && !!app._instance.appState.projections && !!app._instance.appState.projections.temporal,
                selectedCount: app && app._instance && app._instance.appState && app._instance.appState.projections && app._instance.appState.projections.temporal ? app._instance.appState.projections.temporal.selectedDfIds.size : 'no app'
            };
        }
    """
    )
    logger.info(f"JavaScript state: {js_debug}")

    intersection_button.click()
    logger.info("✓ Button clicked")

    # Wait for the POST request to be sent and response received
    logger.info("⏳ Waiting for filter request and response...")

    try:
        # Wait for filtering to complete - use the selected count we detected earlier
        expected_count = selected_count
        expected_percent = f"{expected_count}.0"
        page.wait_for_function(
            f"""
            () => {{
                const status = document.querySelector('.status-info').textContent;
                return status.includes('{expected_count} ({expected_percent}%) of 100 rows');
            }}
            """,
            timeout=5000,
        )

        filtered_status = page.locator(".status-info").text_content() or "NO TEXT CONTENT"
        assert f"{expected_count} ({expected_percent}%) of 100 rows loaded" in filtered_status
    finally:
        current_status = page.locator(".status-info").text_content()
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

        # Check if SSE is working correctly with Vue.js app
        sse_status = page.evaluate(
            """
            () => {
                const app = window.app;
                const appState = app && app._instance && app._instance.appState;
                return {
                    eventSourceExists: !!(appState && appState.eventSource),
                    readyState: appState && appState.eventSource ? appState.eventSource.readyState : 'no eventSource',
                    filterVersion: appState ? appState.filterVersion : 'no appState'
                };
            }
        """
        )

        logger.info(f"SSE Status: {sse_status}")
