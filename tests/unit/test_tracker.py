"""Unit tests for target tracker."""

from __future__ import annotations

from unittest.mock import patch

from acoustic.tracking.doa import MountingOrientation
from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType
from acoustic.tracking.tracker import TargetTracker, TrackedTarget
from acoustic.types import PeakDetection


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
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        # Force last_seen to a known value, then advance time past TTL
        target.last_seen = 100.0

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 106.0  # 6s > 5s TTL
            lost_ids = tracker.tick()

        assert len(lost_ids) == 1

    def test_tick_returns_lost_target_ids(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        tid = target.id
        target.last_seen = 100.0

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 106.0
            lost_ids = tracker.tick()

        assert tid in lost_ids

    def test_get_active_targets_excludes_lost(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        target.last_seen = 100.0

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 106.0
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
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        q.get_nowait()  # drain "new"
        target.last_seen = 100.0

        with patch("acoustic.tracking.tracker.time") as mock_time:
            mock_time.monotonic.return_value = 106.0
            tracker.tick()

        event = q.get_nowait()
        assert event["event"] == "lost"


class TestMultiTargetTracker:
    """Tests for multi-target tracker with nearest-neighbor association (DOA-03)."""

    def test_two_peaks_create_two_targets(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        peaks = [
            PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5),
            PeakDetection(az_deg=-20.0, el_deg=5.0, power=0.8, threshold=0.5),
        ]
        updated = tracker.update_multi(peaks, confidence=0.9)
        assert len(updated) == 2
        assert updated[0].id != updated[1].id
        # Check bearings assigned correctly
        bearings = {(t.az_deg, t.el_deg) for t in updated}
        assert (30.0, 10.0) in bearings
        assert (-20.0, 5.0) in bearings

    def test_existing_target_associates_nearest_peak(self) -> None:
        tracker = TargetTracker(ttl=5.0, association_threshold_deg=7.5)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        t1 = tracker.update_multi(peaks1, confidence=0.9)
        tid = t1[0].id

        # Update with peak at (32, 11) -- within threshold
        peaks2 = [PeakDetection(az_deg=32.0, el_deg=11.0, power=1.0, threshold=0.5)]
        t2 = tracker.update_multi(peaks2, confidence=0.85)
        assert len(t2) == 1
        assert t2[0].id == tid  # same target
        assert t2[0].az_deg == 32.0
        assert t2[0].el_deg == 11.0

    def test_far_peak_creates_new_target(self) -> None:
        tracker = TargetTracker(ttl=5.0, association_threshold_deg=7.5)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks1, confidence=0.9)

        # Update with peak at (60, 20) -- far from existing
        peaks2 = [PeakDetection(az_deg=60.0, el_deg=20.0, power=1.0, threshold=0.5)]
        t2 = tracker.update_multi(peaks2, confidence=0.85)
        assert len(t2) == 1  # only the new one returned from update_multi
        assert len(tracker.get_active_targets()) == 2  # but 2 total active

    def test_unmatched_target_not_immediately_lost(self) -> None:
        tracker = TargetTracker(ttl=5.0, association_threshold_deg=7.5)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        t1 = tracker.update_multi(peaks1, confidence=0.9)
        original_id = t1[0].id

        # Update with peak far away -- original target should NOT be lost
        peaks2 = [PeakDetection(az_deg=60.0, el_deg=20.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks2, confidence=0.85)

        active = tracker.get_active_targets()
        active_ids = {t.id for t in active}
        assert original_id in active_ids  # original still active

    def test_pan_tilt_populated(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        peaks = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        updated = tracker.update_multi(peaks, confidence=0.9)
        assert updated[0].pan_deg == 30.0
        assert updated[0].tilt_deg == 10.0

    def test_broadside_pan_tilt_zero(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        peaks = [PeakDetection(az_deg=0.0, el_deg=0.0, power=1.0, threshold=0.5)]
        updated = tracker.update_multi(peaks, confidence=0.9)
        assert updated[0].pan_deg == 0.0
        assert updated[0].tilt_deg == 0.0

    def test_ema_smoothing_alpha_half(self) -> None:
        tracker = TargetTracker(ttl=5.0, smoothing_alpha=0.5, association_threshold_deg=20.0)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks1, confidence=0.9)

        # Update with peak at (40, 20) -- should blend: 0.5*40 + 0.5*30 = 35
        peaks2 = [PeakDetection(az_deg=40.0, el_deg=20.0, power=1.0, threshold=0.5)]
        updated = tracker.update_multi(peaks2, confidence=0.9)
        assert updated[0].pan_deg == 35.0
        assert updated[0].tilt_deg == 15.0

    def test_ema_smoothing_alpha_one_passthrough(self) -> None:
        tracker = TargetTracker(ttl=5.0, smoothing_alpha=1.0, association_threshold_deg=20.0)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks1, confidence=0.9)

        # Update with peak at (40, 20) -- alpha=1.0 means no smoothing
        peaks2 = [PeakDetection(az_deg=40.0, el_deg=20.0, power=1.0, threshold=0.5)]
        updated = tracker.update_multi(peaks2, confidence=0.9)
        assert updated[0].pan_deg == 40.0
        assert updated[0].tilt_deg == 20.0

    def test_association_threshold_within(self) -> None:
        tracker = TargetTracker(ttl=5.0, association_threshold_deg=7.5)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        t1 = tracker.update_multi(peaks1, confidence=0.9)
        tid = t1[0].id

        # Peak at (37, 10) -- distance 7.0 < 7.5 -- should match
        peaks2 = [PeakDetection(az_deg=37.0, el_deg=10.0, power=1.0, threshold=0.5)]
        t2 = tracker.update_multi(peaks2, confidence=0.9)
        assert t2[0].id == tid

    def test_association_threshold_beyond(self) -> None:
        tracker = TargetTracker(ttl=5.0, association_threshold_deg=7.5)
        # Create target at (30, 10)
        peaks1 = [PeakDetection(az_deg=30.0, el_deg=10.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks1, confidence=0.9)

        # Peak at (38, 10) -- distance 8.0 > 7.5 -- should NOT match
        peaks2 = [PeakDetection(az_deg=38.0, el_deg=10.0, power=1.0, threshold=0.5)]
        tracker.update_multi(peaks2, confidence=0.9)
        assert len(tracker.get_active_targets()) == 2  # two separate targets

    def test_backward_compat_update_still_works(self) -> None:
        tracker = TargetTracker(ttl=5.0)
        target = tracker.update(az_deg=30.0, el_deg=10.0, confidence=0.9)
        assert isinstance(target, TrackedTarget)
        assert target.az_deg == 30.0
