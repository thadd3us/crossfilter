#!/usr/bin/env python3
"""
Script to pre-install Playwright browsers for Docker warm cache.
This script only depends on the uv environment and does not import crossfilter code.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Install Playwright browsers."""
    logger.info("Installing Playwright browsers...")

    try:
        # Install Playwright browsers
        result = subprocess.run(
            # Only shell makes it smaller for CI/CD.
            [sys.executable, "-m", "playwright", "install", "--only-shell"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Playwright browser installation completed successfully")
        logger.info(f"Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Playwright browsers: {e}")
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during Playwright installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
