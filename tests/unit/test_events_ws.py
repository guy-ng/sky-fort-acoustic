"""Unit tests for WebSocket event broadcaster."""

from __future__ import annotations

import asyncio

import pytest

from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType, TargetEvent


def _make_event() -> TargetEvent:
    return TargetEvent(
        event=EventType.NEW,
        target_id="abc-123",
        class_label="drone",
        confidence=0.95,
        az_deg=30.0,
        el_deg=10.0,
        speed_mps=None,
        timestamp=1000.0,
    )


class TestEventBroadcaster:
    def test_subscribe_adds_queue(self) -> None:
        broadcaster = EventBroadcaster()
        q = broadcaster.subscribe()
        assert isinstance(q, asyncio.Queue)
        assert q in broadcaster._subscribers

    def test_unsubscribe_removes_queue(self) -> None:
        broadcaster = EventBroadcaster()
        q = broadcaster.subscribe()
        broadcaster.unsubscribe(q)
        assert q not in broadcaster._subscribers

    def test_broadcast_puts_event_into_all_queues(self) -> None:
        broadcaster = EventBroadcaster()
        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        event = _make_event()
        broadcaster.broadcast(event)

        assert not q1.empty()
        assert not q2.empty()

        msg1 = q1.get_nowait()
        msg2 = q2.get_nowait()
        assert msg1["event"] == "new"
        assert msg2["event"] == "new"

    def test_broadcast_no_subscribers_does_not_raise(self) -> None:
        broadcaster = EventBroadcaster()
        event = _make_event()
        broadcaster.broadcast(event)  # should not raise

    def test_multiple_subscribers_each_receive_same_event(self) -> None:
        broadcaster = EventBroadcaster()
        queues = [broadcaster.subscribe() for _ in range(5)]

        event = _make_event()
        broadcaster.broadcast(event)

        for q in queues:
            msg = q.get_nowait()
            assert msg["target_id"] == "abc-123"
            assert msg["event"] == "new"
