"""Integration tests for recording REST API endpoints."""

from __future__ import annotations

import os
import time

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _use_tmp_data_root(tmp_path, monkeypatch):
    """Override ACOUSTIC_RECORDING_DATA_ROOT so tests write to a temp directory."""
    monkeypatch.setenv("ACOUSTIC_RECORDING_DATA_ROOT", str(tmp_path / "field"))


@pytest.mark.asyncio
class TestRecordingStart:
    """Tests for POST /api/recordings/start."""

    async def test_start_returns_200(self, running_app):
        """POST /api/recordings/start returns 200 with id and status."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/recordings/start")
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["status"] == "recording"
            # Clean up
            await client.post("/api/recordings/stop")

    async def test_start_while_recording_returns_409(self, running_app):
        """POST /api/recordings/start when already recording returns 409."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp1 = await client.post("/api/recordings/start")
            assert resp1.status_code == 200
            resp2 = await client.post("/api/recordings/start")
            assert resp2.status_code == 409
            assert "Already recording" in resp2.json()["message"]
            # Clean up
            await client.post("/api/recordings/stop")


@pytest.mark.asyncio
class TestRecordingStop:
    """Tests for POST /api/recordings/stop."""

    async def test_stop_returns_200(self, running_app):
        """POST /api/recordings/stop after start returns 200 with duration."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            start_resp = await client.post("/api/recordings/start")
            assert start_resp.status_code == 200
            rec_id = start_resp.json()["id"]

            stop_resp = await client.post("/api/recordings/stop")
            assert stop_resp.status_code == 200
            data = stop_resp.json()
            assert data["id"] == rec_id
            assert data["status"] == "stopped"
            assert "duration_s" in data

    async def test_stop_when_not_recording_returns_409(self, running_app):
        """POST /api/recordings/stop when idle returns 409."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/recordings/stop")
            assert response.status_code == 409
            assert "Not recording" in response.json()["message"]


@pytest.mark.asyncio
class TestRecordingList:
    """Tests for GET /api/recordings."""

    async def test_list_recordings(self, running_app):
        """GET /api/recordings after start+stop returns array with 1 item."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            await client.post("/api/recordings/start")
            await client.post("/api/recordings/stop")

            response = await client.get("/api/recordings")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1


@pytest.mark.asyncio
class TestRecordingLabel:
    """Tests for POST /api/recordings/{rec_id}/label."""

    async def test_label_recording(self, running_app):
        """Label a recording with 'drone' and verify it moves."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            start_resp = await client.post("/api/recordings/start")
            rec_id = start_resp.json()["id"]
            await client.post("/api/recordings/stop")

            label_resp = await client.post(
                f"/api/recordings/{rec_id}/label",
                json={"label": "drone"},
            )
            assert label_resp.status_code == 200
            data = label_resp.json()
            assert data["label"] == "drone"
            assert data["directory"] == "drone"

    async def test_invalid_label_returns_400(self, running_app):
        """Label with invalid value returns 400."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            start_resp = await client.post("/api/recordings/start")
            rec_id = start_resp.json()["id"]
            await client.post("/api/recordings/stop")

            label_resp = await client.post(
                f"/api/recordings/{rec_id}/label",
                json={"label": "invalid_label"},
            )
            assert label_resp.status_code == 400


@pytest.mark.asyncio
class TestRecordingUpdate:
    """Tests for PATCH /api/recordings/{rec_id}."""

    async def test_update_metadata(self, running_app):
        """PATCH with notes updates metadata and returns updated record."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            start_resp = await client.post("/api/recordings/start")
            rec_id = start_resp.json()["id"]
            await client.post("/api/recordings/stop")

            # Label first (required for the update to find the file)
            await client.post(
                f"/api/recordings/{rec_id}/label",
                json={"label": "drone"},
            )

            patch_resp = await client.patch(
                f"/api/recordings/{rec_id}",
                json={"notes": "test note"},
            )
            assert patch_resp.status_code == 200
            data = patch_resp.json()
            assert data["notes"] == "test note"


@pytest.mark.asyncio
class TestRecordingDelete:
    """Tests for DELETE /api/recordings/{rec_id}."""

    async def test_delete_recording(self, running_app):
        """DELETE removes recording, then list returns empty."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            start_resp = await client.post("/api/recordings/start")
            rec_id = start_resp.json()["id"]
            await client.post("/api/recordings/stop")

            del_resp = await client.delete(f"/api/recordings/{rec_id}")
            assert del_resp.status_code == 200
            data = del_resp.json()
            assert data["deleted"] is True

            list_resp = await client.get("/api/recordings")
            assert list_resp.status_code == 200
            # The _unlabeled dir may still exist but have no json files
            assert len(list_resp.json()) == 0

    async def test_delete_nonexistent_returns_404(self, running_app):
        """DELETE on nonexistent recording returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.delete("/api/recordings/nonexistent")
            assert response.status_code == 404
