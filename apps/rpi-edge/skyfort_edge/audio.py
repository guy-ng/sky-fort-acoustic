"""Single USB mic capture + 48->32 kHz polyphase resample (D-01/D-02/D-03).

AudioCapture owns a sounddevice.InputStream running a non-allocating PortAudio
callback that writes into a pre-allocated float32 ring buffer. Callers read
most-recent windows on demand via read_window_32k(), which resamples on the
consumer thread (not the audio thread) using scipy.signal.resample_poly(up=2,
down=3).

T-21-13 mitigation: _callback performs only np.copyto + stat updates. It does
not log, does not allocate, and does not call into Python-level logging or
string formatting.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

CAPTURE_SR = 48000
INFERENCE_SR = 32000
RESAMPLE_UP = 2
RESAMPLE_DOWN = 3  # 48000 * 2 / 3 == 32000


@dataclass
class AudioCaptureStats:
    total_callbacks: int = 0
    overruns: int = 0
    last_timestamp: float = 0.0


class AudioCapture:
    """PortAudio InputStream + ring buffer. 1 mono float32 channel at 48 kHz."""

    def __init__(self, device: Optional[str] = None, ring_seconds: float = 4.0) -> None:
        self._device = device
        self._ring_samples = int(ring_seconds * CAPTURE_SR)
        self._ring = np.zeros(self._ring_samples, dtype=np.float32)
        self._write_idx = 0
        self._lock = threading.Lock()
        self._stream = None
        self.stats = AudioCaptureStats()

    def _callback(self, indata, frames, time_info, status):  # pragma: no cover - hw path
        # DO NOT log, do not allocate. Runs on the PortAudio thread (T-21-13).
        if status:
            self.stats.overruns += 1
        mono = indata[:, 0] if indata.ndim == 2 else indata
        with self._lock:
            end = self._write_idx + frames
            if end <= self._ring_samples:
                np.copyto(self._ring[self._write_idx:end], mono)
            else:
                wrap = end - self._ring_samples
                first = frames - wrap
                np.copyto(self._ring[self._write_idx:], mono[:first])
                np.copyto(self._ring[:wrap], mono[first:])
            self._write_idx = end % self._ring_samples
            self.stats.total_callbacks += 1
            try:
                self.stats.last_timestamp = float(time_info.inputBufferAdcTime) if time_info else 0.0
            except Exception:
                self.stats.last_timestamp = 0.0

    def start(self) -> None:  # pragma: no cover - hw path
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=CAPTURE_SR,
            channels=1,
            dtype="float32",
            blocksize=int(0.05 * CAPTURE_SR),  # 50 ms blocks
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        log.info("AudioCapture started: device=%s sr=%d", self._device, CAPTURE_SR)

    def stop(self) -> None:  # pragma: no cover - hw path
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def read_window_48k(self, duration_seconds: float) -> np.ndarray:
        """Copy the most recent ``duration_seconds`` of audio at 48 kHz."""
        n = int(duration_seconds * CAPTURE_SR)
        with self._lock:
            if n > self._ring_samples:
                raise ValueError(
                    f"window {duration_seconds}s exceeds ring {self._ring_samples} samples"
                )
            start = (self._write_idx - n) % self._ring_samples
            if start + n <= self._ring_samples:
                return self._ring[start : start + n].copy()
            head = self._ring[start:].copy()
            tail = self._ring[: n - len(head)].copy()
            return np.concatenate([head, tail])

    def read_window_32k(self, duration_seconds: float) -> np.ndarray:
        """Read most recent window at 48 kHz and resample to 32 kHz."""
        raw = self.read_window_48k(duration_seconds)
        return resample_48k_to_32k(raw)


def resample_48k_to_32k(x: np.ndarray) -> np.ndarray:
    """Polyphase resample 48 kHz -> 32 kHz via scipy.signal.resample_poly(2, 3).

    Standalone helper so tests and callers that do not hold an AudioCapture
    can still get the exact same resampling behavior.
    """
    return resample_poly(x, up=RESAMPLE_UP, down=RESAMPLE_DOWN).astype(np.float32)
