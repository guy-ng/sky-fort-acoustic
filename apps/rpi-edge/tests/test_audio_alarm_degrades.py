"""Wave 0 RED stub — graceful degradation when audio alarm device is unavailable.

Covers: D-19 (audio alarm is best-effort; pipeline must not crash).
Owner: Plan 21-04 (skyfort_edge/alarm.py).
"""
from __future__ import annotations

import pytest


def test_missing_audio_device_logs_warning_and_continues(caplog):
    pytest.fail(
        "not implemented — Plan 21-04 must catch sounddevice errors when no output device "
        "is present, log a warning, and let the pipeline keep running"
    )


def test_playback_failure_does_not_crash_pipeline(caplog):
    pytest.fail(
        "not implemented — Plan 21-04: simulate sd.play() raising; pipeline iteration "
        "must continue without propagating the exception"
    )
