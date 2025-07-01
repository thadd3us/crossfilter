"""Visual regression tests for temporal CDF using syrupy snapshots."""

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright
from syrupy import SnapshotAssertion

from crossfilter.core.data_schema import load_jsonl_to_dataframe
from crossfilter.main import app, get_session_state


@pytest.mark.e2e
async def test_temporal_cdf_visual_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test temporal CDF visualization and capture visual snapshot."""
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
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
        server = uvicorn.Server(config)
        
        # Start server in background
        server_task = asyncio.create_task(server.serve())
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Test with Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1200, "height": 800},
                device_scale_factor=1  # Ensure consistent screenshots
            )
            page = await context.new_page()
            
            # Navigate to the application
            await page.goto("http://localhost:8001/")
            
            # Wait for page to load
            await page.wait_for_selector("h1")
            
            # Check that the page loaded properly
            title = await page.text_content("h1")
            assert "Crossfilter - Temporal CDF Analysis" in title
            
            # Check that status shows data is loaded
            status_text = await page.text_content("#status")
            assert "100 rows" in status_text
            
            # Click refresh plot button to load the CDF
            refresh_btn = page.locator("#refreshBtn")
            await refresh_btn.click()
            
            # Wait for plot to load completely
            await page.wait_for_selector(".js-plotly-plot", timeout=15000)
            
            # Wait for plot to fully render and stabilize
            await asyncio.sleep(3)
            
            # Wait for any animations to complete
            await page.wait_for_function("""
                () => {
                    const plotDiv = document.querySelector('.js-plotly-plot');
                    return plotDiv && plotDiv._fullLayout && plotDiv._fullData;
                }
            """, timeout=10000)
            
            # Take screenshot and compare with snapshot
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # Use syrupy to capture the screenshot as a binary snapshot
            assert screenshot_bytes == snapshot
            
            await browser.close()
            
    finally:
        # Clean up server
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass