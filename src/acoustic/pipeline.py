"""Background beamforming pipeline that consumes audio from a ring buffer."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from acoustic.audio.capture import AudioRingBuffer
from acoustic.beamforming.geometry import build_mic_positions
from acoustic.beamforming.peak import detect_peak_with_threshold
from acoustic.beamforming.srp_phat import srp_phat_2d
from acoustic.config import AcousticSettings
from acoustic.types import PeakDetection, placeholder_target_from_peak

if TYPE_CHECKING:
    from acoustic.classification.state_machine import DetectionStateMachine
    from acoustic.classification.worker import CNNWorker
    from acoustic.tracking.tracker import TargetTracker

logger = logging.getLogger(__name__)


class BeamformingPipeline:
    """Runs SRP-PHAT beamforming on audio chunks from a ring buffer in a background thread."""

    def __init__(
        self,
        settings: AcousticSettings,
        cnn_worker: CNNWorker | None = None,
        state_machine: DetectionStateMachine | None = None,
        tracker: TargetTracker | None = None,
    ) -> None:
        self._settings = settings
        self._mic_positions = build_mic_positions()
        self._az_grid_deg = np.arange(
            -settings.az_range,
            settings.az_range + settings.az_resolution,
            settings.az_resolution,
        )
        self._el_grid_deg = np.arange(
            -settings.el_range,
            settings.el_range + settings.el_resolution,
            settings.el_resolution,
        )
        self.latest_map: np.ndarray | None = None
        self.latest_peak: PeakDetection | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_process_time: float | None = None

        # CNN classification integration (optional)
        self._cnn_worker = cnn_worker
        self._state_machine = state_machine
        self._tracker = tracker
        self._mono_buffer: list[np.ndarray] = []
        self._mono_buffer_samples: int = 0
        self._cnn_segment_samples: int = int(settings.sample_rate * 2.0)

    def process_chunk(self, chunk: np.ndarray) -> PeakDetection | None:
        """Run SRP-PHAT beamforming on a single audio chunk.

        Args:
            chunk: (chunk_samples, num_channels) float32 array from ring buffer.

        Returns:
            PeakDetection if a peak exceeds the noise threshold, else None.
        """
        # Transpose from (samples, channels) to (channels, samples) for SRP-PHAT
        signals = chunk.T

        srp_map = srp_phat_2d(
            signals=signals,
            mic_positions=self._mic_positions,
            fs=self._settings.sample_rate,
            c=self._settings.speed_of_sound,
            az_grid_deg=self._az_grid_deg,
            el_grid_deg=self._el_grid_deg,
            fmin=self._settings.freq_min,
            fmax=self._settings.freq_max,
        )

        self.latest_map = srp_map

        peak = detect_peak_with_threshold(
            srp_map=srp_map,
            az_grid_deg=self._az_grid_deg,
            el_grid_deg=self._el_grid_deg,
            percentile=self._settings.noise_percentile,
            margin=self._settings.noise_margin,
        )
        self.latest_peak = peak
        self._last_process_time = time.monotonic()

        return peak

    def _run_loop(self, ring_buffer: AudioRingBuffer) -> None:
        """Background thread target: continuously read chunks and process them."""
        logger.info("Beamforming pipeline thread started")
        while self._running:
            chunk = ring_buffer.read()
            if chunk is not None:
                try:
                    peak = self.process_chunk(chunk)
                    self._process_cnn(chunk, peak)
                except Exception:
                    logger.exception("Error processing chunk in pipeline")
            else:
                time.sleep(0.01)
        logger.info("Beamforming pipeline thread stopped")

    def _process_cnn(self, chunk: np.ndarray, peak: PeakDetection | None) -> None:
        """Feed audio to CNN worker on peak detection and process results."""
        if self._cnn_worker is None:
            return

        if peak is not None:
            # Accumulate mono audio for CNN segment
            mono = chunk.mean(axis=1).astype(np.float32)
            self._mono_buffer.append(mono)
            self._mono_buffer_samples += len(mono)

            if self._mono_buffer_samples >= self._cnn_segment_samples:
                segment = np.concatenate(self._mono_buffer)[-self._cnn_segment_samples:]
                self._cnn_worker.push(segment, peak.az_deg, peak.el_deg)
                self._mono_buffer.clear()
                self._mono_buffer_samples = 0
        else:
            # No peak -- reset accumulation
            self._mono_buffer.clear()
            self._mono_buffer_samples = 0

        # Check for CNN results
        result = self._cnn_worker.get_latest()
        if result is not None and self._state_machine is not None:
            from acoustic.classification.state_machine import DetectionState

            state = self._state_machine.update(result.drone_probability)
            if state == DetectionState.CONFIRMED and self._tracker is not None:
                self._tracker.update(result.az_deg, result.el_deg, result.drone_probability)

        # Tick tracker for TTL expiry
        if self._tracker is not None:
            self._tracker.tick()

    def start(self, ring_buffer: AudioRingBuffer) -> None:
        """Start the background beamforming thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(ring_buffer,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background beamforming thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def clear_state(self) -> None:
        """Reset pipeline outputs to None (e.g. after device disconnect)."""
        self.latest_map = None
        self.latest_peak = None
        self._last_process_time = None
        logger.info("Pipeline state cleared (device disconnected)")

    def restart(self, ring_buffer: AudioRingBuffer) -> None:
        """Stop, clear state, and restart with a new ring buffer."""
        self.stop()
        self.clear_state()
        self.start(ring_buffer)
        logger.info("Pipeline restarted with new ring buffer")

    @property
    def latest_targets(self) -> list[dict]:
        """Return current target states from tracker, or placeholder fallback."""
        if self._tracker is not None:
            return self._tracker.get_target_states()
        # Fallback when CNN is not available
        peak = self.latest_peak
        if peak is not None:
            return [placeholder_target_from_peak(peak)]
        return []

    @property
    def running(self) -> bool:
        """Whether the pipeline background thread is running."""
        return self._running

    @property
    def last_process_time(self) -> float | None:
        """Monotonic timestamp of the last successful process_chunk call."""
        return self._last_process_time
