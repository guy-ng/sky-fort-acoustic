"""Integration tests for /ws/events and updated /ws/targets endpoints."""

from __future__ import annotations

import threading
import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType, TargetEvent


class TestTargetsWebSocket:
    """Tests for the updated /ws/targets endpoint."""

    def test_targets_returns_list(self, running_app):
        """/ws/targets returns a JSON list."""
        client = TestClient(running_app)
        with client.websocket_connect("/ws/targets") as ws:
            data = ws.receive_json()
            assert isinstance(data, list)

    def test_targets_no_placeholder_import(self):
        """Verify placeholder_target_from_peak is not imported in websocket module."""
        import ast
        from pathlib import Path

        ws_src = Path("src/acoustic/api/websocket.py").read_text()
        tree = ast.parse(ws_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "placeholder_target_from_peak", (
                        "placeholder_target_from_peak should not be imported in websocket.py"
                    )


class TestEventsWebSocket:
    """Tests for the /ws/events endpoint."""

    def test_events_closes_without_broadcaster(self, running_app):
        """/ws/events closes when event_broadcaster is None."""
        original = getattr(running_app.state, "event_broadcaster", None)
        running_app.state.event_broadcaster = None
        try:
            client = TestClient(running_app)
            try:
                with client.websocket_connect("/ws/events") as ws:
                    ws.receive_json()
                    pytest.fail("Expected WebSocket to be closed by server")
            except (WebSocketDisconnect, Exception):
                pass  # Expected -- server closed with 1011
        finally:
            running_app.state.event_broadcaster = original

    def test_events_accepts_and_receives_event(self, running_app):
        """/ws/events accepts connection and delivers broadcast events."""
        broadcaster = EventBroadcaster()
        original = getattr(running_app.state, "event_broadcaster", None)
        running_app.state.event_broadcaster = broadcaster
        try:
            client = TestClient(running_app)
            with client.websocket_connect("/ws/events") as ws:
                event = TargetEvent(
                    event=EventType.NEW,
                    target_id="test-id-1234",
                    class_label="drone",
                    confidence=0.95,
                    az_deg=10.0,
                    el_deg=5.0,
                    speed_mps=None,
                    timestamp=time.monotonic(),
                )

                def send_delayed():
                    time.sleep(0.3)
                    broadcaster.broadcast(event)

                t = threading.Thread(target=send_delayed)
                t.start()

                data = ws.receive_json(mode="text")
                t.join()

                assert data["event"] == "new"
                assert data["target_id"] == "test-id-1234"
                assert data["class_label"] == "drone"
                assert data["confidence"] == 0.95
        finally:
            running_app.state.event_broadcaster = original

    def test_events_receives_ordered_events(self, running_app):
        """Multiple events are received in order by /ws/events client."""
        broadcaster = EventBroadcaster()
        original = getattr(running_app.state, "event_broadcaster", None)
        running_app.state.event_broadcaster = broadcaster
        try:
            client = TestClient(running_app)
            with client.websocket_connect("/ws/events") as ws:

                def send_events():
                    time.sleep(0.3)
                    for etype in [EventType.NEW, EventType.UPDATE, EventType.LOST]:
                        event = TargetEvent(
                            event=etype,
                            target_id="test-multi",
                            class_label="drone",
                            confidence=0.8,
                            az_deg=20.0,
                            el_deg=10.0,
                            timestamp=time.monotonic(),
                        )
                        broadcaster.broadcast(event)

                t = threading.Thread(target=send_events)
                t.start()

                received_types = []
                for _ in range(3):
                    data = ws.receive_json(mode="text")
                    received_types.append(data["event"])
                t.join()

                assert received_types == ["new", "update", "lost"]
        finally:
            running_app.state.event_broadcaster = original
