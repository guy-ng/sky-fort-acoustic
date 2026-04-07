"""Audio capture pipeline: ring buffer and callback-based InputStream."""

from __future__ import annotations

import logging
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

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
        on_stream_finished: callable | None = None,
        mic_channels: tuple[int, ...] | None = None,
        highpass_hz: float | None = None,
        lowpass_hz: float | None = None,
    ) -> None:
        # When `mic_channels` is given, open the device with enough channels
        # to cover the highest requested index but only forward those channels
        # into the ring buffer. Used to skip on-board DSP channels (e.g.
        # ReSpeaker XMOS firmware: channel 0 is the AGC/AEC/beamformer output,
        # channels 1-4 are raw mic capsules).
        if mic_channels is not None:
            stream_channels = max(mic_channels) + 1
            ring_channel_count = len(mic_channels)
        else:
            stream_channels = channels
            ring_channel_count = channels

        self._mic_channels = mic_channels
        self._ring = AudioRingBuffer(
            num_chunks=ring_chunks,
            chunk_samples=chunk_samples,
            num_channels=ring_channel_count,
        )

        # Optional streaming Butterworth filter applied per ring channel
        # before writing into the buffer. When both highpass_hz and lowpass_hz
        # are set we build a single bandpass SOS rather than cascading two
        # filters — same response, half the per-callback work and half the
        # state. State is preserved across callbacks so consecutive chunks
        # produce a continuous filtered signal without transient artifacts
        # at chunk boundaries.
        self._hp_sos: np.ndarray | None = None
        self._hp_zi: np.ndarray | None = None
        nyq = fs / 2.0

        def _norm(hz: float | None) -> float | None:
            if hz is None or hz <= 0:
                return None
            wn = float(hz) / nyq
            return wn if 0 < wn < 1 else None

        wn_hp = _norm(highpass_hz)
        wn_lp = _norm(lowpass_hz)

        sos = None
        label = None
        if wn_hp is not None and wn_lp is not None and wn_hp < wn_lp:
            sos = butter(4, [wn_hp, wn_lp], btype="bandpass", output="sos")
            label = f"bandpass {highpass_hz:.0f}-{lowpass_hz:.0f} Hz"
        elif wn_hp is not None:
            sos = butter(4, wn_hp, btype="highpass", output="sos")
            label = f"high-pass {highpass_hz:.0f} Hz"
        elif wn_lp is not None:
            sos = butter(4, wn_lp, btype="lowpass", output="sos")
            label = f"low-pass {lowpass_hz:.0f} Hz"

        if sos is not None:
            self._hp_sos = sos
            zi = sosfilt_zi(sos)  # (n_sections, 2)
            self._hp_zi = np.repeat(
                zi[np.newaxis, :, :], ring_channel_count, axis=0
            )
            logger.info("Capture filter enabled: %s Butterworth (order 4)", label)
        elif highpass_hz or lowpass_hz:
            logger.warning(
                "Ignoring filter (hp=%s, lp=%s) — outside valid range for fs=%d",
                highpass_hz,
                lowpass_hz,
                fs,
            )
        self._last_frame_time: float | None = None
        self._xrun_flag = False
        self._on_stream_finished = on_stream_finished

        self._stream = sd.InputStream(
            device=device,
            samplerate=fs,
            channels=stream_channels,
            dtype="float32",
            blocksize=chunk_samples,
            callback=self._callback,
            finished_callback=self._finished,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback -- minimal work only, no logging or GIL-heavy ops."""
        self._last_frame_time = time.monotonic()
        if status:
            self._xrun_flag = True

        # Channel selection
        if self._mic_channels is not None:
            if len(self._mic_channels) == 1:
                ch = self._mic_channels[0]
                data = indata[:, ch : ch + 1]
            else:
                data = indata[:, list(self._mic_channels)]
        else:
            data = indata

        # Optional high-pass (state preserved across callbacks)
        if self._hp_sos is not None:
            filtered = np.empty_like(data)
            for ch in range(data.shape[1]):
                filtered[:, ch], self._hp_zi[ch] = sosfilt(
                    self._hp_sos, data[:, ch], zi=self._hp_zi[ch]
                )
            data = filtered

        self._ring.write(data)

    def _finished(self) -> None:
        """Called by PortAudio when the stream stops (e.g. device unplugged)."""
        if self._on_stream_finished is not None:
            self._on_stream_finished()

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
