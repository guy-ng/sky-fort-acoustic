"""Unit tests for target tracker."""

from __future__ import annotations

from unittest.mock import patch

from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType
from acoustic.tracking.tracker import TargetTracker, TrackedTarget


class TestTrackerCreate:
    def test_update_confirmed_creates_new_target(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        assert isinstance(target, TrackedTarget)
        assert len(target.id) == 36  # UUID4 string length
        assert target.class_label == "drone"
        assert target.az_deg == 30.0
        assert target.el_deg == 10.0
        assert target.confidence == 0.9
        assert target.speed_mps is None

    def test_update_existing_target_updates_bearing(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        t1 = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        t2 = tracker.update(az_deg=45.0, el_deg=15.0, confidence=0.85)
        assert t1.id == t2.id  # same UUID
        assert t2.az_deg == 45.0
        assert t2.el_deg == 15.0
        assert t2.confidence == 0.85

    def test_update_returns_tracked_target_with_none_speed(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=0.0, el_deg=0.0, confidence=0.5)
        assert target.speed_mps is None

    def test_multiple_updates_keep_same_uuid(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        t1 = tracker.update(az_deg=10.0, el_deg=5.0, confidence=0.9)
        t2 = tracker.update(az_deg=20.0, el_deg=8.0, confidence=0.8)
        t3 = tracker.update(az_deg=30.0, el_deg=12.0, confidence=0.7)
        assert t1.id == t2.id == t3.id


class TestTrackerTTL:
    def test_target_lost_after_ttl(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)

        # Simulate time passing beyond TTL
        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 99999.0
            lost_ids = tracker.tick()

        assert len(lost_ids) == 1

    def test_tick_returns_lost_target_ids(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        tid = target.id

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 99999.0
            lost_ids = tracker.tick()

        assert tid in lost_ids

    def test_get_active_targets_excludes_lost(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 99999.0
            tracker.tick()

        assert len(tracker.get_active_targets()) == 0


class TestTrackerState:
    def test_get_active_targets_returns_non_lost(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        active = tracker.get_active_targets()
        assert len(active) == 1
        assert active[0].lost is False

    def test_get_target_states_returns_dicts(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        states = tracker.get_target_states()
        assert len(states) == 1
        s = states[0]
        assert "id" in s
        assert "class_label" in s
        assert "speed_mps" in s
        assert s["speed_mps"] is None
        assert "az_deg" in s
        assert "el_deg" in s
        assert "confidence" in s


class TestTrackerEvents:
    def test_new_target_emits_new_event(self) -> None:
        broadcaster = EventBroadcaster()
        import asyncio
        q = asyncio.Queue()
        broadcaster._subscribers.add(q)

        tracker = TargetTracker(ttl=5.0, broadcaster=broadcaster)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)

        event = q.get_nowait()
        assert event["event"] == "new"

    def test_update_target_emits_update_event(self) -> None:
        broadcaster = EventBroadcaster()
        import asyncio
        q = asyncio.Queue()
        broadcaster._subscribers.add(q)

        tracker = TargetTracker(ttl=5.0, broadcaster=broadcaster)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        # Drain the "new" event
        q.get_nowait()

        tracker.update(az_deg=45.0, el_deg=15.0, confidence=0.85)
        event = q.get_nowait()
        assert event["event"] == "update"

    def test_lost_target_emits_lost_event(self) -> None:
        broadcaster = EventBroadcaster()
        import asyncio
        q = asyncio.Queue()
        broadcaster._subscribers.add(q)

        tracker = TargetTracker(ttl=5.0, broadcaster=broadcaster)
        tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        q.get_nowait()  # drain "new"

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 99999.0
            tracker.tick()

        event = q.get_nowait()
        assert event["event"] == "lost"
