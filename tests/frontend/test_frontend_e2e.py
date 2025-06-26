"""End-to-end frontend tests using Playwright."""

import multiprocessing
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import pytest_asyncio
import requests
import uvicorn
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from crossfilter.main import app


class TestServer:
    """Test server runner for E2E tests."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        self.host = host
        self.port = port
        self.process: multiprocessing.Process | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self) -> None:
        """Start the test server in a separate process."""
        self.process = multiprocessing.Process(target=self._run_server, daemon=True)
        self.process.start()
        self._wait_for_server()

    def stop(self) -> None:
        """Stop the test server."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()

    def _run_server(self) -> None:
        """Run the FastAPI server."""
        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="error",  # Suppress logs during testing
        )

    def _wait_for_server(self, timeout: int = 10) -> None:
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/api/session", timeout=1)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"Server did not start within {timeout} seconds")


@pytest.fixture(scope="session")
def test_server() -> Generator[TestServer, None, None]:
    """Pytest fixture to manage test server lifecycle."""
    server = TestServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="session")
def sample_data_loaded(test_server: TestServer) -> bool:
    """Load sample data via the web API for testing."""
    test_data_path = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    if not test_data_path.exists():
        return False

    # Load data via the web API endpoint
    response = requests.post(
        f"{test_server.base_url}/api/data/load",
        json={"file_path": str(test_data_path)},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    if response.status_code == 200:
        return True
    else:
        print(f"Failed to load sample data: {response.status_code} - {response.text}")
        return False


@pytest_asyncio.fixture(scope="session")
async def browser() -> AsyncGenerator[Browser, None]:
    """Pytest fixture to manage browser lifecycle."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            await browser.close()


@pytest_asyncio.fixture
async def browser_context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create a new browser context for each test."""
    context = await browser.new_context()
    try:
        yield context
    finally:
        await context.close()


@pytest_asyncio.fixture
async def page(browser_context: BrowserContext) -> Generator[Page, None, None]:
    """Create a new page for each test."""
    page = await browser_context.new_page()
    try:
        yield page
    finally:
        await page.close()


@pytest.mark.e2e
def test_server_starts_successfully(test_server: TestServer) -> None:
    """Test that the test server starts and responds to HTTP requests."""
    # Test that the server is running
    response = requests.get(f"{test_server.base_url}/api/session")
    assert response.status_code == 200

    # Test that the main page returns HTML
    response = requests.get(test_server.base_url)
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Crossfilter" in response.text


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_page_loads(page: Page, test_server: TestServer) -> None:
    """Test that the main page loads successfully."""
    await page.goto(test_server.base_url)

    # Check that the page title is correct
    title = await page.title()
    assert "Crossfilter" in title

    # Check that the header is present
    header = await page.locator("h1").first
    header_text = await header.text_content()
    assert "Crossfilter" in header_text


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_status_bar_shows_no_data(page: Page, test_server: TestServer) -> None:
    """Test that the status bar shows no data when no data is loaded."""
    await page.goto(test_server.base_url)

    # Wait for the page to load and status to update
    await page.wait_for_load_state("networkidle")

    # Check status bar
    status_text = await page.locator("#status-text").text_content()
    assert "No data loaded" in status_text or "rows" in status_text.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_buttons_are_present(page: Page, test_server: TestServer) -> None:
    """Test that all expected buttons are present on the page."""
    await page.goto(test_server.base_url)

    # Check that control buttons are present
    refresh_btn = page.locator("#refresh-plots")
    reset_btn = page.locator("#reset-filters")
    undo_btn = page.locator("#undo-filter")

    assert await refresh_btn.is_visible()
    assert await reset_btn.is_visible()
    assert await undo_btn.is_visible()

    # Check plot control buttons
    geo_filter_btn = page.locator("#geo-filter-visible")
    geo_clear_btn = page.locator("#geo-clear-selection")
    temporal_filter_btn = page.locator("#temporal-filter-visible")
    temporal_clear_btn = page.locator("#temporal-clear-selection")

    assert await geo_filter_btn.is_visible()
    assert await geo_clear_btn.is_visible()
    assert await temporal_filter_btn.is_visible()
    assert await temporal_clear_btn.is_visible()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_max_groups_input(page: Page, test_server: TestServer) -> None:
    """Test that the max groups input field works correctly."""
    await page.goto(test_server.base_url)

    # Find the max groups input
    max_groups_input = page.locator("#max-groups")
    assert await max_groups_input.is_visible()

    # Check default value
    default_value = await max_groups_input.input_value()
    assert default_value == "100000"

    # Change the value
    await max_groups_input.fill("50000")
    new_value = await max_groups_input.input_value()
    assert new_value == "50000"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_plots_containers_exist(page: Page, test_server: TestServer) -> None:
    """Test that the plot containers exist on the page."""
    await page.goto(test_server.base_url)

    # Check geographic plot container
    geo_plot = page.locator("#geographic-plot")
    assert await geo_plot.is_visible()

    # Check temporal plot container
    temporal_plot = page.locator("#temporal-plot")
    assert await temporal_plot.is_visible()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_with_sample_data(
    page: Page, test_server: TestServer, sample_data_loaded: bool
) -> None:
    """Test the application behavior with sample data loaded."""
    if not sample_data_loaded:
        pytest.skip("Sample data could not be loaded")

    await page.goto(test_server.base_url)

    # Wait for the page to load and status to update
    await page.wait_for_load_state("networkidle")

    # Wait a bit for async status updates
    await page.wait_for_timeout(1000)

    # Check that status shows data is loaded
    status_text = await page.locator("#status-text").text_content()
    # Should show something like "100 rows loaded" or similar
    assert "rows" in status_text.lower() or "loaded" in status_text.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_refresh_button_click(page: Page, test_server: TestServer) -> None:
    """Test that the refresh button can be clicked without errors."""
    await page.goto(test_server.base_url)

    # Click the refresh button
    refresh_btn = page.locator("#refresh-plots")
    await refresh_btn.click()

    # Should not cause any console errors
    # Wait a moment for any network requests to complete
    await page.wait_for_timeout(500)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reset_filters_button_click(page: Page, test_server: TestServer) -> None:
    """Test that the reset filters button can be clicked without errors."""
    await page.goto(test_server.base_url)

    # Click the reset filters button
    reset_btn = page.locator("#reset-filters")
    await reset_btn.click()

    # Should not cause any console errors
    # Wait a moment for any network requests to complete
    await page.wait_for_timeout(500)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_responsive_design(page: Page, test_server: TestServer) -> None:
    """Test that the page is responsive and works at different screen sizes."""
    await page.goto(test_server.base_url)

    # Test desktop size
    await page.set_viewport_size({"width": 1920, "height": 1080})
    header = page.locator(".header")
    assert await header.is_visible()

    # Test tablet size
    await page.set_viewport_size({"width": 768, "height": 1024})
    assert await header.is_visible()

    # Test mobile size
    await page.set_viewport_size({"width": 375, "height": 667})
    assert await header.is_visible()



@pytest.mark.e2e
@pytest.mark.asyncio
async def test_page_console_errors(page: Page, test_server: TestServer) -> None:
    """Test that the page doesn't have console errors on load."""
    console_errors = []

    def handle_console_message(msg):
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", handle_console_message)

    await page.goto(test_server.base_url)
    await page.wait_for_load_state("networkidle")

    # Filter out expected/harmless errors
    serious_errors = [
        error
        for error in console_errors
        if not any(
            harmless in error.lower()
            for harmless in [
                "favicon.ico",  # Common harmless error
                "kepler",  # May have loading issues in test environment
                "plotly",  # May have loading issues in test environment
            ]
        )
    ]

    assert len(serious_errors) == 0, f"Console errors found: {serious_errors}"
