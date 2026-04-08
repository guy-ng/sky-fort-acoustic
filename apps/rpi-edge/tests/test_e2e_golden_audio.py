"""Wave 0 RED stub — end-to-end pipeline test on golden WAV fixtures.

Covers: D-03 (full pipeline), D-12/D-14 (hysteresis), D-20 (detection log), with mocked GPIO.
Owner: Plan 21-05 (integration entry point) + Plan 21-04 (sub-components).
"""
from __future__ import annotations

import pytest


def test_golden_drone_wav_produces_latched_detection(golden_drone_wav, mock_gpio_factory, tmp_jsonl_log):
    pytest.fail(
        "not implemented — Plan 21-05: feed golden_drone_1s_48k.wav through full pipeline "
        "(resample -> mel -> ONNX -> hysteresis -> GPIO -> log); assert at least one "
        "latched detection record in tmp_jsonl_log and LED pin asserted high"
    )


def test_golden_silence_wav_produces_no_detection(golden_silence_wav, mock_gpio_factory, tmp_jsonl_log):
    pytest.fail(
        "not implemented — Plan 21-05: feed golden_silence_1s_48k.wav and assert zero "
        "detection records and LED pin remains low"
    )
