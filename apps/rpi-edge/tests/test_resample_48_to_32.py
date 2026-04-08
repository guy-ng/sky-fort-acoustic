"""D-02: scipy.signal.resample_poly(up=2, down=3) 48->32 kHz correctness + latency.

Owner: Plan 21-05 (skyfort_edge/audio.py).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from skyfort_edge.audio import resample_48k_to_32k


def test_resample_poly_2_3_correctness():
    # 1 s at 48 kHz -> 32000 samples at 32 kHz
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    y = resample_48k_to_32k(x)
    assert len(y) == 32000
    assert y.dtype == np.float32
    # Silence in -> silence out
    zeros = np.zeros(48000, dtype=np.float32)
    assert np.max(np.abs(resample_48k_to_32k(zeros))) < 1e-6

    # Sinusoid preserves its frequency after resampling: a 1 kHz tone at 48 kHz
    # must still read as ~1 kHz at 32 kHz. Check via FFT peak bin.
    t_48 = np.arange(48000, dtype=np.float32) / 48000.0
    tone_48 = np.sin(2 * np.pi * 1000.0 * t_48).astype(np.float32)
    tone_32 = resample_48k_to_32k(tone_48)
    assert len(tone_32) == 32000
    spectrum = np.abs(np.fft.rfft(tone_32))
    peak_bin = int(np.argmax(spectrum))
    peak_freq = peak_bin * 32000.0 / 32000.0  # rfft on 32000 samples -> 1 Hz/bin
    assert abs(peak_freq - 1000.0) < 2.0, f"expected ~1000 Hz, got {peak_freq}"


def test_resample_latency_under_50ms_for_1s_window():
    x = np.random.default_rng(1).standard_normal(48000).astype(np.float32)
    # Warmup to prime any caches
    resample_48k_to_32k(x)
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        resample_48k_to_32k(x)
    elapsed_per_call = (time.perf_counter() - start) / iterations
    assert (
        elapsed_per_call < 0.05
    ), f"resample latency {elapsed_per_call * 1000:.1f} ms >= 50 ms budget"
