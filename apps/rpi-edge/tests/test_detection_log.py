"""Wave 0 RED stub — rotating JSONL detection log.

Covers: D-20 (one record per latch), D-21 (size-based rotation), D-22 (cannot disable).
Owner: Plan 21-04 (skyfort_edge/detection_log.py).
"""
from __future__ import annotations

import pytest


def test_one_jsonl_record_per_latch(tmp_jsonl_log):
    pytest.fail(
        "not implemented — Plan 21-04: drive hysteresis to LATCHED state once and assert "
        "exactly one JSONL record appears in the log"
    )


def test_required_fields_present(tmp_jsonl_log):
    pytest.fail(
        "not implemented — Plan 21-04: each record must contain {timestamp, probability, "
        "duration_s, model_version, ...} (per D-20)"
    )


def test_size_based_rotation(tmp_jsonl_log):
    pytest.fail(
        "not implemented — Plan 21-04: write past max_bytes and assert log rotates to "
        "detections.jsonl.1 etc."
    )


def test_cannot_be_disabled_via_config(tmp_config_dir):
    pytest.fail(
        "not implemented — Plan 21-04: attempting to set detection_log.enabled=false in "
        "YAML must raise / be ignored (D-22 forensic requirement)"
    )
