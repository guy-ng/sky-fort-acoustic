"""Background pipeline that consumes audio from a ring buffer.

Beamforming (SRP-PHAT) is currently stubbed out — the pipeline produces a
zero map and no peaks.  The beamforming module is kept in the codebase for
future re-integration.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from acoustic.audio.capture import AudioRingBuffer
from acoustic.config import AcousticSettings
from acoustic.types import PeakDetection, placeholder_target_from_peak

if TYPE_CHECKING:
    from acoustic.classification.state_machine import DetectionStateMachine
    from acoustic.classification.worker import CNNWorker
    from acoustic.tracking.tracker import TargetTracker

logger = logging.getLogger(__name__)

MAX_DETECTION_LOG = 200


@dataclass
class DetectionLogEntry:
    """A single detection event in the log."""

    timestamp: float
    drone_probability: float
    detection_state: str
    message: str


@dataclass
class DetectionSession:
    """Active detection session configuration."""

    model_path: str
    confidence: float = 0.90
    time_frame: float = 2.0
    positive_detections: int = 2
    gain: float = 3.0
    log: deque = field(default_factory=lambda: deque(maxlen=MAX_DETECTION_LOG))


class BeamformingPipeline:
    """Consumes audio chunks from a ring buffer in a background thread.

    Beamforming is currently stubbed — process_chunk returns a zero map and
    no peak.  CNN classification still runs on the raw audio.
    """

    def __init__(
        self,
        settings: AcousticSettings,
        cnn_worker: CNNWorker | None = None,
        state_machine: DetectionStateMachine | None = None,
        tracker: TargetTracker | None = None,
        recording_manager: object | None = None,
    ) -> None:
        self._settings = settings
        self._az_size = int((2 * settings.az_range / settings.az_resolution) + 1)
        self._el_size = int((2 * settings.el_range / settings.el_resolution) + 1)
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
        self._cnn_segment_samples: int = int(settings.sample_rate * 0.5)
        self._default_segment_samples: int = self._cnn_segment_samples
        self._last_cnn_push: float = 0.0
        self._cnn_interval: float = 0.25  # Push every 0.25s for 50% overlap of 0.5s segments
        self._default_cnn_interval: float = self._cnn_interval
        self._last_cnn_result_ts: float = 0.0  # Dedup CNN results by timestamp

        # Recording integration (passive observer -- receives chunks from pipeline)
        self._recording_manager = recording_manager

        # Detection session (user-initiated via Pipeline tab)
        self._detection_session: DetectionSession | None = None
        self._detection_lock = threading.Lock()

    def process_chunk(self, chunk: np.ndarray) -> PeakDetection | None:
        """Stub: produce a zero beamforming map with no peak detection.

        Args:
            chunk: (chunk_samples, num_channels) float32 array from ring buffer.

        Returns:
            Always None (no beamforming peak).
        """
        self.latest_map = np.zeros((self._az_size, self._el_size), dtype=np.float32)
        self.latest_peak = None
        self._last_process_time = time.monotonic()
        return None

    def _run_loop(self, ring_buffer: AudioRingBuffer) -> None:
        """Background thread target: continuously read chunks and process them."""
        logger.info("Beamforming pipeline thread started")
        while self._running:
            chunk = ring_buffer.read()
            if chunk is not None:
                try:
                    # Forward to recording manager (passive observer)
                    if self._recording_manager is not None:
                        self._recording_manager.feed_chunk(chunk)
                    peak = self.process_chunk(chunk)
                    self._process_cnn(chunk, peak)
                except Exception:
                    logger.exception("Error processing chunk in pipeline")
            else:
                time.sleep(0.01)
        logger.info("Beamforming pipeline thread stopped")

    def start_detection_session(
        self,
        model_path: str,
        confidence: float = 0.90,
        time_frame: float = 2.0,
        positive_detections: int = 2,
        gain: float = 3.0,
        model_type: str = "research_cnn",
    ) -> None:
        """Start a user-initiated detection session with custom parameters."""
        with self._detection_lock:
            self._detection_session = DetectionSession(
                model_path=model_path,
                confidence=confidence,
                time_frame=time_frame,
                positive_detections=positive_detections,
                gain=gain,
            )

        # EfficientAT needs 1s segments at 48kHz (resampled to 32kHz by preprocessor)
        # with 0.25s overlap (push every 0.75s)
        if "efficientat" in model_type or "mn10" in model_type or "mn05" in model_type:
            self._cnn_segment_samples = int(self._settings.sample_rate * 1.0)  # 1s
            self._cnn_interval = 0.75  # 0.25s overlap
        else:
            self._cnn_segment_samples = self._default_segment_samples
            self._cnn_interval = self._default_cnn_interval

        # Reset mono buffer for new segment size
        self._mono_buffer.clear()
        self._mono_buffer_samples = 0

        # Reconfigure state machine thresholds
        if self._state_machine is not None:
            self._state_machine._enter = confidence
            self._state_machine._exit = max(0.1, confidence - 0.4)
            self._state_machine._confirm = positive_detections
            self._state_machine.reset()
        logger.info(
            "Detection session started: model=%s type=%s segment=%.1fs interval=%.2fs confidence=%.0f%% gain=%.1f",
            model_path, model_type, self._cnn_segment_samples / self._settings.sample_rate,
            self._cnn_interval, confidence * 100, gain,
        )

    def stop_detection_session(self) -> None:
        """Stop the current detection session."""
        with self._detection_lock:
            self._detection_session = None
        # Restore default segment config
        self._cnn_segment_samples = self._default_segment_samples
        self._cnn_interval = self._default_cnn_interval
        self._mono_buffer.clear()
        self._mono_buffer_samples = 0
        if self._state_machine is not None:
            self._state_machine.reset()
        if self._tracker is not None:
            self._tracker.clear()
        logger.info("Detection session stopped")

    @property
    def detection_session(self) -> DetectionSession | None:
        """Return the current detection session, or None."""
        with self._detection_lock:
            return self._detection_session

    def _process_cnn(self, chunk: np.ndarray, peak: PeakDetection | None) -> None:
        """Feed audio to CNN worker on peak detection and process results."""
        if self._cnn_worker is None:
            return

        session = self._detection_session

        # Always accumulate mono audio so we have a rolling buffer ready
        mono = chunk.mean(axis=1).astype(np.float32)

        # Apply gain if detection session is active
        if session is not None and session.gain != 1.0:
            mono = mono * session.gain
        self._mono_buffer.append(mono)
        self._mono_buffer_samples += len(mono)

        # Trim buffer to at most 2x the segment length to bound memory
        max_samples = self._cnn_segment_samples * 2
        while self._mono_buffer_samples > max_samples:
            dropped = self._mono_buffer.pop(0)
            self._mono_buffer_samples -= len(dropped)

        # Push to CNN when we have enough audio accumulated:
        # - On peak: push immediately with peak bearing
        # - No peak: push periodically (every _cnn_interval) with (0,0) bearing
        #   so the UI always has a fresh drone probability
        now = time.monotonic()
        if self._mono_buffer_samples >= self._cnn_segment_samples:
            segment = np.concatenate(self._mono_buffer)[-self._cnn_segment_samples:]
            if peak is not None:
                self._cnn_worker.push(segment, peak.az_deg, peak.el_deg)
                self._last_cnn_push = now
            elif now - self._last_cnn_push >= self._cnn_interval:
                self._cnn_worker.push(segment, 0.0, 0.0)
                self._last_cnn_push = now

        # Check for CNN results (dedup by timestamp)
        result = self._cnn_worker.get_latest()
        if result is not None and result.timestamp != self._last_cnn_result_ts and self._state_machine is not None:
            self._last_cnn_result_ts = result.timestamp
            from acoustic.classification.state_machine import DetectionState

            prev_state = self._state_machine.state
            state = self._state_machine.update(result.drone_probability)
            if state == DetectionState.CONFIRMED and self._tracker is not None:
                self._tracker.update(result.az_deg, result.el_deg, result.drone_probability)

            # Log every CNN result for the pipeline tab
            if session is not None:
                now = time.time()
                prob = result.drone_probability
                if state != prev_state:
                    msg = f"{prev_state.value} \u2192 {state.value} (prob={prob:.1%})"
                else:
                    msg = f"[{state.value}] prob={prob:.1%}"
                session.log.append(DetectionLogEntry(
                    timestamp=now,
                    drone_probability=prob,
                    detection_state=state.value,
                    message=msg,
                ))

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
    def latest_drone_probability(self) -> float | None:
        """Return the most recent CNN drone probability, regardless of detection state."""
        if self._cnn_worker is None:
            return None
        result = self._cnn_worker.get_latest()
        return result.drone_probability if result is not None else None

    @property
    def latest_detection_state(self) -> str | None:
        """Return the current detection state machine label."""
        if self._state_machine is None:
            return None
        return self._state_machine.state.value

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
