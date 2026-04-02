"""Integration tests for training WebSocket endpoint."""

from __future__ import annotations

from starlette.testclient import TestClient


class TestTrainingWebSocket:
    """Tests for WS /ws/training."""

    def test_training_ws_sends_initial_status(self, running_app):
        """Connect to /ws/training receives initial status JSON with status field."""
        client = TestClient(running_app)
        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_json()
            assert "status" in data
            assert data["status"] in ("idle", "running", "completed", "failed", "cancelled")
