"""Integration tests for WebSocket endpoints."""

from __future__ import annotations

import numpy as np
from starlette.testclient import TestClient


class TestHeatmapWebSocket:
    """Tests for WS /ws/heatmap."""

    def test_heatmap_ws_sends_handshake(self, running_app):
        """First message on /ws/heatmap is a JSON handshake with grid dimensions."""
        client = TestClient(running_app)
        with client.websocket_connect("/ws/heatmap") as ws:
            data = ws.receive_json()
            assert data["type"] == "handshake"
            assert "width" in data
            assert "height" in data
            assert "az_min" in data
            assert "az_max" in data

    def test_heatmap_ws_sends_binary(self, running_app):
        """After handshake, receives binary float32 frame with correct size."""
        client = TestClient(running_app)
        with client.websocket_connect("/ws/heatmap") as ws:
            handshake = ws.receive_json()
            width = handshake["width"]
            height = handshake["height"]
            data = ws.receive_bytes()
            values = np.frombuffer(data, dtype=np.float32)
            assert len(values) == width * height


class TestTargetsWebSocket:
    """Tests for WS /ws/targets."""

    def test_targets_ws_sends_json(self, running_app):
        """First message on /ws/targets is a JSON array."""
        client = TestClient(running_app)
        with client.websocket_connect("/ws/targets") as ws:
            data = ws.receive_json()
            assert isinstance(data, list)
