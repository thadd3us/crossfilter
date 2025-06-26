"""Hello World frontend test using Playwright."""

from pathlib import Path

import pytest
from playwright.sync_api import Page


@pytest.mark.e2e
def test_hello_world_page(page: Page) -> None:
    """Test that the hello world page loads and displays correct content."""
    # Get the path to the static HTML file
    hello_html_path = Path(__file__).parent / "test_data" / "hello.html"
    file_url = f"file://{hello_html_path.absolute()}"

    page.goto(file_url)

    # Check the page title
    title = page.title()
    assert title == "Hello World"

    # Check the main heading content
    heading = page.locator("h1").text_content()
    assert heading == "Hello World!"
