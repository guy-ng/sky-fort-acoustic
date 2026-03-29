"""Integration tests for the /health endpoint."""

from __future__ import annotations

import asyncio
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure simulated mode for tests (no hardware needed)
os.environ["ACOUSTIC_AUDIO_SOURCE"] = "simulated"

from acoustic.main import app, lifespan  # noqa: E402


@pytest.fixture
async def running_app():
    """Start the app lifespan (audio capture + pipeline) and yield the app."""
    async with lifespan(app):
        # Give pipeline a moment to start processing
        await asyncio.sleep(0.3)
        yield app


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
