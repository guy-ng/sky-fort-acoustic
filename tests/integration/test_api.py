"""Integration tests for REST API endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
class TestMapEndpoint:
    """Tests for GET /api/map."""

    async def test_map_returns_200(self, running_app):
        """GET /api/map returns 200 with beamforming map data."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/map")
            assert response.status_code == 200

    async def test_map_has_grid_metadata(self, running_app):
        """Response contains all grid metadata fields."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/map")
            data = response.json()
            for key in ("az_min", "az_max", "el_min", "el_max", "width", "height"):
                assert key in data, f"Missing key: {key}"

    async def test_map_data_is_2d_array(self, running_app):
        """Data field is a 2D array with correct dimensions."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/map")
            data = response.json()
            grid = data["data"]
            assert isinstance(grid, list)
            assert len(grid) == data["height"]
            assert all(len(row) == data["width"] for row in grid)

    async def test_map_peak_present_when_detected(self, running_app):
        """Peak field is present (may be null or dict with az/el/power)."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/map")
            data = response.json()
            assert "peak" in data
            if data["peak"] is not None:
                for key in ("az_deg", "el_deg", "power"):
                    assert key in data["peak"]


@pytest.mark.asyncio
class TestTargetsEndpoint:
    """Tests for GET /api/targets."""

    async def test_targets_returns_200(self, running_app):
        """GET /api/targets returns 200."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/targets")
            assert response.status_code == 200

    async def test_targets_returns_list(self, running_app):
        """Response is a JSON array (may be empty or have 1 placeholder target)."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/targets")
            data = response.json()
            assert isinstance(data, list)
