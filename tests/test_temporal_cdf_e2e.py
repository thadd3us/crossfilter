"""End-to-end tests for temporal CDF functionality."""

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from crossfilter.core.data_schema import load_jsonl_to_dataframe
from crossfilter.main import app, get_session_state


@pytest.mark.e2e
async def test_temporal_cdf_with_sample_data(tmp_path: Path) -> None:
    """Test temporal CDF visualization with sample data."""
    # Load sample data into session state
    sample_data_path = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    assert sample_data_path.exists(), f"Sample data not found at {sample_data_path}"
    
    # Load data into the global session state
    df = load_jsonl_to_dataframe(sample_data_path)
    session_state = get_session_state()
    session_state.load_dataframe(df)
    
    # Start the server in a separate task
    import uvicorn
    server_task = None
    
    try:
        # Create server config
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning")
        server = uvicorn.Server(config)
        
        # Start server in background
        server_task = asyncio.create_task(server.serve())
        
        # Wait a bit for server to start
        await asyncio.sleep(1)
        
        # Test with Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1200, "height": 800})
            page = await context.new_page()
            
            # Navigate to the application
            await page.goto("http://localhost:8001/")
            
            # Wait for page to load
            await page.wait_for_selector("h1")
            
            # Check that the page title is correct
            title = await page.text_content("h1")
            assert "Crossfilter - Temporal CDF Analysis" in title
            
            # Check that status shows data is loaded
            status_text = await page.text_content("#status")
            assert "100 rows" in status_text
            
            # Click refresh plot button to load the CDF
            refresh_btn = page.locator("#refreshBtn")
            await refresh_btn.click()
            
            # Wait for plot to load
            await page.wait_for_selector(".js-plotly-plot", timeout=10000)
            
            # Wait a bit more for plot to fully render
            await asyncio.sleep(2)
            
            # Take screenshot for visual verification
            screenshot_path = tmp_path / "temporal_cdf_screenshot.png"
            await page.screenshot(path=str(screenshot_path))
            
            # Verify screenshot was created
            assert screenshot_path.exists()
            assert screenshot_path.stat().st_size > 1000  # Reasonable file size
            
            # Test plot interaction - check that the plot has data
            plot_element = page.locator(".js-plotly-plot")
            assert await plot_element.is_visible()
            
            # Check that the plot has the expected structure
            # Look for SVG elements that indicate a rendered plot
            svg_elements = page.locator(".js-plotly-plot svg")
            svg_count = await svg_elements.count()
            assert svg_count > 0, "Plot should contain SVG elements"
            
            await browser.close()
            
    finally:
        # Clean up server
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


@pytest.mark.e2e
async def test_temporal_cdf_plot_selection(tmp_path: Path) -> None:
    """Test plot selection functionality."""
    # Load sample data
    sample_data_path = Path(__file__).parent.parent / "test_data" / "sample_100.jsonl"
    df = load_jsonl_to_dataframe(sample_data_path)
    session_state = get_session_state()
    session_state.load_dataframe(df)
    
    server_task = None
    
    try:
        # Start server
        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(1)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1200, "height": 800})
            page = await context.new_page()
            
            await page.goto("http://localhost:8001/")
            await page.wait_for_selector("h1")
            
            # Load the plot
            refresh_btn = page.locator("#refreshBtn")
            await refresh_btn.click()
            await page.wait_for_selector(".js-plotly-plot", timeout=10000)
            await asyncio.sleep(2)
            
            # Test reset filters functionality
            reset_btn = page.locator("#resetFiltersBtn")
            await reset_btn.click()
            
            # Check that reset was successful by looking for success message
            await page.wait_for_timeout(1000)  # Wait for response
            
            # Verify that the filtered count matches total count after reset
            status_text = await page.text_content("#status")
            assert "100 after filtering" in status_text or "100 rows" in status_text
            
            await browser.close()
            
    finally:
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    # Run the test standalone for debugging
    asyncio.run(test_temporal_cdf_with_sample_data(Path("/tmp")))