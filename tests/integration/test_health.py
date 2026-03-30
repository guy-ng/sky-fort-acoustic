"""Integration tests for the /health endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_200(self, running_app):
        """GET /health returns 200 status."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200

    async def test_health_json_fields(self, running_app):
        """Response JSON contains all required keys."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            data = response.json()
            required_keys = {
                "status",
                "device_detected",
                "pipeline_running",
                "overflow_count",
                "last_frame_time",
            }
            assert required_keys.issubset(
                data.keys()
            ), f"Missing keys: {required_keys - data.keys()}"

    async def test_health_simulated_mode(self, running_app):
        """In simulated mode, device_detected=false and pipeline_running=true."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            data = response.json()
            assert data["device_detected"] is False
            assert data["pipeline_running"] is True

    async def test_health_status_ok(self, running_app):
        """Response has status='ok' when pipeline is running."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            data = response.json()
            assert data["status"] == "ok"
