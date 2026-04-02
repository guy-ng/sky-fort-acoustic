"""Integration tests for training REST API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
class TestTrainingStart:
    """Tests for POST /api/training/start."""

    async def test_start_returns_200(self, running_app):
        """POST /api/training/start with defaults returns 200 with message."""
        # Mock manager.start to avoid actually training
        manager = running_app.state.training_manager
        with patch.object(manager, "start"):
            async with AsyncClient(
                transport=ASGITransport(app=running_app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.post("/api/training/start", json={})
                assert response.status_code == 200
                data = response.json()
                assert "Training started" in data["message"]

    async def test_start_returns_409_when_running(self, running_app):
        """POST /api/training/start when already running returns 409."""
        manager = running_app.state.training_manager
        with patch.object(manager, "is_training", return_value=True):
            async with AsyncClient(
                transport=ASGITransport(app=running_app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.post("/api/training/start", json={})
                assert response.status_code == 409
                data = response.json()
                assert "already in progress" in data["message"]


@pytest.mark.asyncio
class TestTrainingProgress:
    """Tests for GET /api/training/progress."""

    async def test_progress_returns_200(self, running_app):
        """GET /api/training/progress returns 200 with status field."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/training/progress")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] in ("idle", "running", "completed", "cancelled", "failed")


@pytest.mark.asyncio
class TestTrainingCancel:
    """Tests for POST /api/training/cancel."""

    async def test_cancel_when_not_running_returns_409(self, running_app):
        """POST /api/training/cancel when idle returns 409."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/training/cancel")
            assert response.status_code == 409
            data = response.json()
            assert "No training is currently running" in data["message"]
