"""Unit tests for demand-driven beamforming gate (BF-16).

Tests the gate logic that activates beamforming only when CNN detects a drone,
holds for bf_holdoff_seconds after last detection, and returns zero map when idle.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from acoustic.config import AcousticSettings
from acoustic.pipeline import BeamformingPipeline
from acoustic.types import PeakDetection


class FakeStateMachine:
    """Minimal mock of DetectionStateMachine with configurable state."""

    def __init__(self, state_value: str = "NO_DRONE") -> None:
        from acoustic.classification.state_machine import DetectionState

        self._state = DetectionState(state_value)

    @property
    def state(self):
        return self._state

    def set_state(self, value: str) -> None:
        from acoustic.classification.state_machine import DetectionState

        self._state = DetectionState(value)


def _make_pipeline(
    state_machine=None,
    settings: AcousticSettings | None = None,
) -> BeamformingPipeline:
    """Create a pipeline with optional mock state machine."""
    s = settings or AcousticSettings()
    return BeamformingPipeline(settings=s, state_machine=state_machine)


def _make_chunk(
    n_samples: int = 7200,
    n_channels: int = 16,
    freq: float = 2000.0,
    fs: int = 48000,
) -> np.ndarray:
    """Synthetic multi-channel audio chunk with a tone."""
    t = np.arange(n_samples) / fs
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    chunk = np.tile(tone[:, np.newaxis], (1, n_channels))
    chunk += np.random.default_rng(42).standard_normal(chunk.shape).astype(np.float32) * 0.01
    return chunk


class TestGateOff:
    """Test 1: When state is NO_DRONE and holdoff expired, return zero map."""

    def test_returns_empty_list_when_no_drone_and_holdoff_expired(self) -> None:
        sm = FakeStateMachine("NO_DRONE")
        pipe = _make_pipeline(state_machine=sm)
        # Ensure holdoff is expired (last_bf_active_time = 0, way in the past)
        pipe._last_bf_active_time = 0.0
        chunk = _make_chunk()
        peaks = pipe.process_chunk(chunk)
        assert peaks == []
        assert pipe.latest_map is not None
        assert np.all(pipe.latest_map == 0)


class TestGateOn:
    """Test 2: When state is CONFIRMED, beamforming runs and returns peaks."""

    def test_returns_peaks_when_confirmed(self) -> None:
        sm = FakeStateMachine("DRONE_CONFIRMED")
        pipe = _make_pipeline(state_machine=sm)
        chunk = _make_chunk()
        peaks = pipe.process_chunk(chunk)
        assert isinstance(peaks, list)
        assert pipe.latest_map is not None
        assert not np.all(pipe.latest_map == 0)


class TestHoldoff:
    """Test 3 & 4: Holdoff timer logic."""

    def test_beamforming_continues_during_holdoff(self, monkeypatch) -> None:
        """After state goes from CONFIRMED to NO_DRONE, BF continues for holdoff."""
        sm = FakeStateMachine("NO_DRONE")
        pipe = _make_pipeline(state_machine=sm)
        # Simulate: last CONFIRMED was 4.9s ago (within 5.0s holdoff)
        fake_now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        pipe._last_bf_active_time = fake_now - 4.9
        chunk = _make_chunk()
        peaks = pipe.process_chunk(chunk)
        # Should still run beamforming (within holdoff)
        assert pipe.latest_map is not None
        assert not np.all(pipe.latest_map == 0)

    def test_beamforming_stops_after_holdoff(self, monkeypatch) -> None:
        """After holdoff expires (5.1s), BF returns zero map."""
        sm = FakeStateMachine("NO_DRONE")
        pipe = _make_pipeline(state_machine=sm)
        fake_now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        pipe._last_bf_active_time = fake_now - 5.1
        chunk = _make_chunk()
        peaks = pipe.process_chunk(chunk)
        assert peaks == []
        assert pipe.latest_map is not None
        assert np.all(pipe.latest_map == 0)


class TestNoStateMachine:
    """Test 5: Without state machine, beamforming always runs (backward compat)."""

    def test_always_runs_without_state_machine(self) -> None:
        pipe = _make_pipeline(state_machine=None)
        chunk = _make_chunk()
        peaks = pipe.process_chunk(chunk)
        assert isinstance(peaks, list)
        assert pipe.latest_map is not None
        assert not np.all(pipe.latest_map == 0)


class TestLatestPeaks:
    """Test 6: latest_peaks property."""

    def test_latest_peaks_is_list(self) -> None:
        pipe = _make_pipeline(state_machine=None)
        chunk = _make_chunk()
        pipe.process_chunk(chunk)
        assert isinstance(pipe.latest_peaks, list)

    def test_latest_peak_returns_first_or_none(self) -> None:
        pipe = _make_pipeline(state_machine=None)
        chunk = _make_chunk()
        pipe.process_chunk(chunk)
        if pipe.latest_peaks:
            assert pipe.latest_peak == pipe.latest_peaks[0]
        else:
            assert pipe.latest_peak is None
