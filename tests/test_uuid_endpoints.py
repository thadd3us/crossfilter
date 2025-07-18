"""Tests for UUID-related API endpoints."""

import logging

import requests

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)


def test_active_uuids_endpoint_no_data():
    """Test the active UUIDs endpoint with no data loaded via TestClient."""
    from fastapi.testclient import TestClient

    from crossfilter.main import app

    with TestClient(app) as client:
        response = client.get("/api/active_uuids")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["uuids"] == ""


def test_active_uuids_endpoint_with_server_data(server_with_data: str):
    """Test the active UUIDs endpoint with sample data loaded from server."""
    server_url = server_with_data

    # First verify that data is loaded
    response = requests.get(f"{server_url}/api/session")
    assert response.status_code == 200
    session_data = response.json()
    assert session_data["has_data"] is True
    assert session_data["row_count"] == 100

    # Test the UUID endpoint
    response = requests.get(f"{server_url}/api/active_uuids")
    assert response.status_code == 200

    data = response.json()
    assert "count" in data
    assert "uuids" in data
    assert isinstance(data["count"], int)
    assert isinstance(data["uuids"], str)

    # The sample data should contain some image UUIDs
    assert data["count"] > 0, "Expected to find some image UUIDs in sample data"

    # Verify UUID string format
    if data["count"] > 0:
        uuid_list = data["uuids"].split(",")
        assert len(uuid_list) == data["count"], "UUID count should match number of UUIDs in string"

        # Check that all UUIDs are non-empty
        for uuid in uuid_list:
            assert uuid.strip(), "All UUIDs should be non-empty"

    logger.info(f"Found {data['count']} active image UUIDs")
    logger.info(f"First few UUIDs: {data['uuids'][:100]}...")


def test_active_uuids_endpoint_with_limit(server_with_data: str):
    """Test the active UUIDs endpoint with limit parameter."""
    server_url = server_with_data

    # Test with limit of 5
    response = requests.get(f"{server_url}/api/active_uuids?limit=5")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] <= 5, "Should respect the limit parameter"

    if data["count"] > 0:
        uuid_list = data["uuids"].split(",")
        assert len(uuid_list) == data["count"]


def test_active_uuids_endpoint_limit_validation(server_with_data: str):
    """Test the active UUIDs endpoint with invalid limit parameters."""
    server_url = server_with_data

    # Test limit too small
    response = requests.get(f"{server_url}/api/active_uuids?limit=0")
    assert response.status_code == 422  # Validation error

    # Test limit too large
    response = requests.get(f"{server_url}/api/active_uuids?limit=10001")
    assert response.status_code == 422  # Validation error

    # Test valid limits
    response = requests.get(f"{server_url}/api/active_uuids?limit=1")
    assert response.status_code == 200

    response = requests.get(f"{server_url}/api/active_uuids?limit=10000")
    assert response.status_code == 200
