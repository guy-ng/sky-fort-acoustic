"""D-19: AudioAlarm silent-degrade semantics."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from skyfort_edge.audio_alarm import AudioAlarm

REPO_ROOT = Path(__file__).resolve().parents[1]
ALERT_WAV = REPO_ROOT / "assets/alert.wav"


def test_missing_audio_device_logs_warning_and_continues(caplog, monkeypatch):
    caplog.set_level(logging.WARNING, logger="skyfort_edge.audio_alarm")
    alarm = AudioAlarm(
        enabled=True,
        alert_wav_path=ALERT_WAV,
        device="nonexistent_device_xyz",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("PortAudio error: device not found")

    import sounddevice as sd

    monkeypatch.setattr(sd, "play", boom)

    # Must not raise
    alarm.play()
    assert any(
        "degraded silently" in r.message or "play failed" in r.message
        for r in caplog.records
    )


def test_playback_failure_does_not_crash_pipeline(monkeypatch):
    alarm = AudioAlarm(enabled=True, alert_wav_path=ALERT_WAV)
    import sounddevice as sd

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sd, "play", boom)
    alarm.play()  # must not raise


def test_disabled_alarm_is_noop():
    alarm = AudioAlarm(enabled=False, alert_wav_path=ALERT_WAV)
    alarm.play()  # no-op, no exception


def test_play_once_per_latch_cycle(monkeypatch):
    """D-18: After play(), subsequent play() calls are no-ops until reset() is called."""
    alarm = AudioAlarm(enabled=True, alert_wav_path=ALERT_WAV)
    call_count = [0]
    import sounddevice as sd

    def count(*args, **kwargs):
        call_count[0] += 1

    monkeypatch.setattr(sd, "play", count)
    alarm.play()
    alarm.play()
    alarm.play()
    assert call_count[0] == 1
    alarm.reset()
    alarm.play()
    assert call_count[0] == 2
