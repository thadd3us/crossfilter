#!/usr/bin/env python3
"""
Simple script to run tests in parallel.

This script demonstrates the parallel testing capabilities configured for the project.
Uses pytest-xdist to run tests across multiple CPU cores for faster execution.
"""
import subprocess
import sys

def main():
    """Run tests in parallel with nice output."""
    print("Running tests in parallel...")
    print("=" * 60)
    
    # Run the fast tests (excluding resource-intensive and e2e tests)
    cmd = [
        sys.executable, "-m", "pytest",
        "-m", "not (resource_intensive or e2e)",
        "--tb=short",
        "-v"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())