"""D-20/D-21/D-22: Always-on rotating JSONL detection log tests."""
from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path

import pytest

from skyfort_edge.config import DetectionLogConfig
from skyfort_edge.detection_log import DetectionLogger
from skyfort_edge.hysteresis import EventType, StateEvent


@pytest.fixture
def log_cfg(tmp_path):
    return DetectionLogConfig(
        path=str(tmp_path / "detections.jsonl"),
        rotate_max_bytes=10 * 1024 * 1024,
        rotate_backup_count=3,
    )


def test_one_jsonl_record_per_latch(log_cfg):
    dl = DetectionLogger(log_cfg)
    for i in range(5):
        ev = StateEvent(
            EventType.RISING_EDGE, float(i), 0.9, latch_duration_seconds=2.0
        )
        dl.write_latch(ev, class_name="drone", score=0.9)
    dl.close()
    lines = Path(log_cfg.path).read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # must parse


def test_required_fields_present(log_cfg):
    dl = DetectionLogger(log_cfg)
    ev = StateEvent(
        EventType.FALLING_EDGE, 1.0, 0.85, latch_duration_seconds=3.5
    )
    dl.write_latch(ev, class_name="drone", score=0.85)
    dl.close()
    record = json.loads(Path(log_cfg.path).read_text().strip().splitlines()[0])
    assert "timestamp" in record
    assert "class" in record and record["class"] == "drone"
    assert "score" in record and record["score"] == pytest.approx(0.85)
    assert "latch_duration_seconds" in record and record[
        "latch_duration_seconds"
    ] == pytest.approx(3.5)


def test_size_based_rotation(tmp_path):
    cfg = DetectionLogConfig(
        path=str(tmp_path / "detections.jsonl"),
        rotate_max_bytes=200,  # tiny → forces rotation
        rotate_backup_count=3,
    )
    dl = DetectionLogger(cfg)
    for i in range(30):
        ev = StateEvent(
            EventType.RISING_EDGE, float(i), 0.9, latch_duration_seconds=2.0
        )
        dl.write_latch(
            ev, class_name="drone_class_with_padding_xxxxx", score=0.9
        )
    dl.close()
    backups = list(tmp_path.glob("detections.jsonl.*"))
    assert len(backups) >= 1, f"expected rotation backups, got {backups}"


def test_cannot_be_disabled_via_config():
    """D-21: DetectionLogConfig must NOT have an 'enabled' or 'disabled' field."""
    field_names = {f.name for f in fields(DetectionLogConfig)}
    assert "enabled" not in field_names
    assert "disabled" not in field_names


def test_uses_dedicated_logger_with_propagate_false(log_cfg):
    dl = DetectionLogger(log_cfg)
    assert dl.logger.name == "skyfort_edge.detection"
    assert dl.logger.propagate is False
    dl.close()


def test_general_log_level_does_not_affect_detection_log(log_cfg):
    """D-21: silencing the root logger must not silence the detection log."""
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        dl = DetectionLogger(log_cfg)
        ev = StateEvent(
            EventType.RISING_EDGE, 0.0, 0.9, latch_duration_seconds=2.0
        )
        dl.write_latch(ev, class_name="drone", score=0.9)
        dl.close()
        assert Path(log_cfg.path).read_text().strip() != ""
    finally:
        logging.getLogger().setLevel(logging.WARNING)  # restore
