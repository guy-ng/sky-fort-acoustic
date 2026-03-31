"""Unit tests for target event schema."""

from __future__ import annotations

from acoustic.tracking.schema import EventType, TargetEvent


class TestEventType:
    def test_new_value(self) -> None:
        assert EventType.NEW == "new"
        assert EventType.NEW.value == "new"

    def test_update_value(self) -> None:
        assert EventType.UPDATE == "update"
        assert EventType.UPDATE.value == "update"

    def test_lost_value(self) -> None:
        assert EventType.LOST == "lost"
        assert EventType.LOST.value == "lost"


class TestTargetEvent:
    def _make_event(self, **overrides) -> TargetEvent:
        defaults = {
            "event": EventType.NEW,
            "target_id": "abc-123",
            "class_label": "drone",
            "confidence": 0.95,
            "az_deg": 30.0,
            "el_deg": 10.0,
            "speed_mps": None,
            "timestamp": 1000.0,
        }
        defaults.update(overrides)
        return TargetEvent(**defaults)

    def test_validates_with_all_fields(self) -> None:
        event = self._make_event()
        assert event.target_id == "abc-123"
        assert event.class_label == "drone"
        assert event.confidence == 0.95

    def test_model_dump_is_json_serializable(self) -> None:
        event = self._make_event()
        d = event.model_dump()
        assert isinstance(d, dict)
        assert d["event"] == "new"
        assert d["target_id"] == "abc-123"

    def test_speed_mps_accepts_none(self) -> None:
        event = self._make_event(speed_mps=None)
        assert event.speed_mps is None

    def test_timestamp_is_float(self) -> None:
        event = self._make_event(timestamp=12345.678)
        assert isinstance(event.timestamp, float)

    def test_new_event_serializes(self) -> None:
        event = self._make_event(event=EventType.NEW)
        d = event.model_dump()
        assert d["event"] == "new"

    def test_lost_event_serializes(self) -> None:
        event = self._make_event(event=EventType.LOST)
        d = event.model_dump()
        assert d["event"] == "lost"
