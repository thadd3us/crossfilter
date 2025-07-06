"""Tests for UUID preview images endpoint."""

import logging
from pathlib import Path

import requests
from syrupy import SnapshotAssertion

from tests.fixtures_server import server_with_data

assert server_with_data, "Don't remove this import!"

logger = logging.getLogger(__name__)


def test_get_uuid_preview_image_existing(server_with_data: str) -> None:
    """Test getting a preview image for an existing UUID."""
    server_url = server_with_data
    
    # Test with an existing UUID - we created this in the test data
    uuid = "123456789abcdef0123456789abcdef0"
    
    response = requests.get(f"{server_url}/api/image_preview/uuid/{uuid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify we got actual image data
    assert len(response.content) > 1000  # Should be a reasonable size for a JPEG
    
    # Verify it starts with JPEG magic bytes
    assert response.content[:2] == b'\xff\xd8'


def test_get_uuid_preview_image_nonexistent(server_with_data: str) -> None:
    """Test getting a preview image for a nonexistent UUID."""
    server_url = server_with_data
    
    # Test with a UUID that doesn't exist
    uuid = "nonexistent123456789abcdef0123"
    
    response = requests.get(f"{server_url}/api/image_preview/uuid/{uuid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    
    # Verify we got the dummy SVG image
    assert b"No preview available" in response.content
    assert b"<svg" in response.content


def test_get_uuid_preview_image_short_uuid(server_with_data: str) -> None:
    """Test getting a preview image for a UUID that's too short."""
    server_url = server_with_data
    
    # Test with a UUID that's too short (less than 2 characters)
    uuid = "x"
    
    response = requests.get(f"{server_url}/api/image_preview/uuid/{uuid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    
    # Verify we got the dummy SVG image
    assert b"No preview available" in response.content


def test_get_uuid_preview_image_different_subdirs(server_with_data: str) -> None:
    """Test getting preview images from different subdirectories."""
    server_url = server_with_data
    
    # Test different UUID prefixes that should exist
    test_cases = [
        ("123456789abcdef0123456789abcdef0", "image/jpeg"),  # 12/ subdir
        ("3456789abcdef0123456789abcdef01", "image/jpeg"),   # 34/ subdir  
        ("56789abcdef0123456789abcdef012", "image/jpeg"),    # 56/ subdir
        ("789abcdef0123456789abcdef0123", "image/jpeg"),     # 78/ subdir
        ("90abcdef0123456789abcdef012345", "image/jpeg"),    # 90/ subdir
        ("ab123456789abcdef0123456789ab", "image/svg+xml"),  # ab/ subdir (empty)
        ("cd123456789abcdef0123456789cd", "image/svg+xml"),  # cd/ subdir (empty)
    ]
    
    for uuid, expected_content_type in test_cases:
        response = requests.get(f"{server_url}/api/image_preview/uuid/{uuid}")
        assert response.status_code == 200
        assert response.headers["content-type"] == expected_content_type
        
        if expected_content_type == "image/jpeg":
            # Should be actual JPEG image
            assert len(response.content) > 1000
            assert response.content[:2] == b'\xff\xd8'
        else:
            # Should be dummy SVG
            assert b"No preview available" in response.content






def test_get_uuid_preview_image_without_base_dir(source_tree_root: Path) -> None:
    """Test the endpoint when no UUID preview images base directory is configured."""
    from tests.fixtures_server import server_with_data
    import subprocess
    import sys
    import time
    from pathlib import Path
    
    # Start a server without the UUID preview images directory
    sample_data_path = source_tree_root / "test_data" / "sample_100.jsonl"
    test_port = 8002
    server_url = f"http://localhost:{test_port}"
    
    cmd = (
        sys.executable,
        *("-m", "crossfilter.main", "serve"),
        *("--port", str(test_port)),
        *("--preload_jsonl", str(sample_data_path)),
        # Note: NOT including --uuid_preview_images_base_dir
    )
    
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=source_tree_root,
    )
    
    try:
        # Wait for server to start with exponential backoff
        def wait_for_server_with_backoff(server_url: str, max_attempts: int = 15) -> bool:
            delay_ms = 1
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{server_url}/api/session", timeout=1)
                    if response.status_code == 200:
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(delay_ms / 1000)  # Convert ms to seconds
                delay_ms = min(delay_ms * 2, 1000)  # Cap at 1 second: 1ms, 2ms, 4ms, ..., 1000ms
            return False
        
        # Wait for server to be ready
        if not wait_for_server_with_backoff(server_url):
            raise TimeoutError("Server failed to start within timeout")
        
        # Test that we get dummy images for any UUID
        response = requests.get(f"{server_url}/api/image_preview/uuid/123456789abcdef0123456789abcdef0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert b"No preview available" in response.content
        
    finally:
        server_process.kill()
        server_process.wait()


def test_uuid_preview_image_endpoint_response_format(
    server_with_data: str, snapshot: SnapshotAssertion
) -> None:
    """Test that the UUID preview image endpoint response headers match expectations."""
    server_url = server_with_data
    
    # Test with existing image
    uuid = "123456789abcdef0123456789abcdef0"
    response = requests.get(f"{server_url}/api/image_preview/uuid/{uuid}")
    
    # Check response structure (not content, since that's binary)
    response_info = {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "content_length": len(response.content),
        "has_jpeg_magic": response.content[:2] == b'\xff\xd8',
    }
    
    assert response_info == snapshot
    
    # Test with nonexistent image
    uuid_nonexistent = "nonexistent123456789abcdef0123"
    response_dummy = requests.get(f"{server_url}/api/image_preview/uuid/{uuid_nonexistent}")
    
    response_dummy_info = {
        "status_code": response_dummy.status_code,
        "content_type": response_dummy.headers.get("content-type"),
        "content_contains_svg": b"<svg" in response_dummy.content,
        "content_contains_no_preview": b"No preview available" in response_dummy.content,
    }
    
    assert response_dummy_info == snapshot