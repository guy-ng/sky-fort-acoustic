"""Integration tests for the /health endpoint."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure simulated mode for tests (no hardware needed)
os.environ["ACOUSTIC_AUDIO_SOURCE"] = "simulated"

from acoustic.main import app  # noqa: E402


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_200(self):
        """GET /health returns 200 status."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200

    async def test_health_json_fields(self):
        """Response JSON contains all required keys."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
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
            assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - data.keys()}"

    async def test_health_simulated_mode(self):
        """In simulated mode, device_detected=false and pipeline_running=true."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            data = response.json()
            assert data["device_detected"] is False
            assert data["pipeline_running"] is True

    async def test_health_status_ok(self):
        """Response has status='ok' when pipeline is running."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            data = response.json()
            assert data["status"] == "ok"
