"""Global pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def source_tree_root() -> Path:
    """Return the root of the source tree."""
    # Get the directory of this conftest.py file, which should be at the root
    return Path(__file__).parent


@pytest.fixture
def test_catalogs_dir(source_tree_root: Path) -> Path:
    """Return the path to the test Lightroom catalogs directory."""
    return source_tree_root / "test_data" / "lightroom" / "test_catalogs"
