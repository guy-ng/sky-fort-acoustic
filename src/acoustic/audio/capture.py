"""Audio capture pipeline: ring buffer and callback-based InputStream."""

from __future__ import annotations

import logging
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """Lock-free circular buffer for 16-channel audio chunks.

    Pre-allocates a contiguous NumPy array and uses write/read indices
    for FIFO access. One slot is reserved to distinguish full from empty.
    """

    def __init__(self, num_chunks: int, chunk_samples: int, num_channels: int) -> None:
        self._buffer = np.zeros((num_chunks, chunk_samples, num_channels), dtype=np.float32)
        self._num_chunks = num_chunks
        self._write_idx = 0
        self._read_idx = 0
        self._overflow_count = 0

    def write(self, data: np.ndarray) -> bool:
        """Write a chunk into the buffer. Returns False if full (overflow)."""
        next_idx = (self._write_idx + 1) % self._num_chunks
        if next_idx == self._read_idx:
            self._overflow_count += 1
            return False
        np.copyto(self._buffer[self._write_idx], data)
        self._write_idx = next_idx
        return True

    def read(self) -> np.ndarray | None:
        """Read the oldest chunk from the buffer. Returns None if empty."""
        if self._read_idx == self._write_idx:
            return None
        data = self._buffer[self._read_idx].copy()
        self._read_idx = (self._read_idx + 1) % self._num_chunks
        return data

    @property
    def available(self) -> int:
        """Number of chunks available to read."""
        return (self._write_idx - self._read_idx) % self._num_chunks

    @property
    def overflow_count(self) -> int:
        """Number of overflow events (writes that were dropped)."""
        return self._overflow_count


class AudioCapture:
    """Callback-based audio capture using sounddevice.InputStream.

    Writes incoming audio chunks into an AudioRingBuffer. The callback
    does minimal work (np.copyto only) to avoid blocking the audio thread.
    """

    def __init__(
        self,
        device: str | int | None,
        fs: int,
        channels: int,
        chunk_samples: int,
        ring_chunks: int = 14,
    ) -> None:
        self._ring = AudioRingBuffer(
            num_chunks=ring_chunks,
            chunk_samples=chunk_samples,
            num_channels=channels,
        )
        self._last_frame_time: float | None = None
        self._xrun_flag = False

        self._stream = sd.InputStream(
            device=device,
            samplerate=fs,
            channels=channels,
            dtype="float32",
            blocksize=chunk_samples,
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback -- minimal work only, no logging or GIL-heavy ops."""
        self._last_frame_time = time.monotonic()
        if status:
            self._xrun_flag = True
        self._ring.write(indata)

    def start(self) -> None:
        """Start the audio stream."""
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio stream.

        Tolerates PortAudioError -- the USB device may already be gone.
        """
        try:
            self._stream.stop()
        except (sd.PortAudioError, Exception) as exc:
            logger.warning("Stream stop failed (device may be disconnected): %s", exc)

        try:
            self._stream.close()
        except (sd.PortAudioError, Exception) as exc:
            logger.warning("Stream close failed (device may be disconnected): %s", exc)

    @property
    def ring(self) -> AudioRingBuffer:
        """Access the underlying ring buffer."""
        return self._ring

    @property
    def last_frame_time(self) -> float | None:
        """Monotonic timestamp of the last received audio frame."""
        return self._last_frame_time
