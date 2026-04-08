"""Shared fixtures for apps/rpi-edge test suite (Wave 0 stubs).

Provides:
- golden_drone_wav / golden_silence_wav: deterministic 1s@48kHz audio fixtures
- mock_gpio_factory: gpiozero MockFactory for SIGTERM / LED tests
- tmp_config_dir: empty temp dir for config-merge tests
- tmp_jsonl_log: temp path for rotating detection-log tests
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_drone_wav():
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(FIXTURES / "golden_drone_1s_48k.wav")
    return audio.astype(np.float32), sr


@pytest.fixture
def golden_silence_wav():
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(FIXTURES / "golden_silence_1s_48k.wav")
    return audio.astype(np.float32), sr


@pytest.fixture
def mock_gpio_factory():
    gpiozero = pytest.importorskip("gpiozero")
    from gpiozero.pins.mock import MockFactory

    prev = gpiozero.Device.pin_factory
    gpiozero.Device.pin_factory = MockFactory()
    yield gpiozero.Device.pin_factory
    gpiozero.Device.pin_factory = prev


@pytest.fixture
def tmp_config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def tmp_jsonl_log(tmp_path):
    return tmp_path / "detections.jsonl"
