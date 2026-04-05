"""Background pipeline that consumes audio from a ring buffer.

Runs real SRP-PHAT beamforming with bandpass pre-filtering (BF-11),
MCRA adaptive noise estimation (BF-14), multi-peak detection (BF-13),
and parabolic interpolation for sub-degree accuracy (BF-12).

Beamforming is demand-driven (BF-16): it activates when the CNN state
machine reports DRONE_CONFIRMED and holds for bf_holdoff_seconds after
the last detection, then idles to save compute.
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
from acoustic.beamforming import (
    BandpassFilter,
    MCRANoiseEstimator,
    build_mic_positions,
    detect_multi_peak,
    parabolic_interpolation_2d,
    srp_phat_2d,
)
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

    Runs real SRP-PHAT beamforming with bandpass pre-filter, MCRA noise
    estimation, multi-peak detection, and parabolic interpolation.
    Beamforming is demand-driven: gated by CNN detection state with holdoff.
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

        # Beamforming engine components (BF-10 through BF-16)
        self._mic_positions = build_mic_positions()
        self._bandpass = BandpassFilter(
            settings.sample_rate,
            settings.bf_freq_min,
            settings.bf_freq_max,
            settings.bf_filter_order,
        )
        self._mcra = MCRANoiseEstimator(
            alpha_s=settings.bf_mcra_alpha_s,
            alpha_d=settings.bf_mcra_alpha_d,
            delta=settings.bf_mcra_delta,
            min_window=settings.bf_mcra_min_window,
        )
        self._az_grid = np.arange(
            -settings.az_range,
            settings.az_range + settings.az_resolution,
            settings.az_resolution,
        )
        self._el_grid = np.arange(
            -settings.el_range,
            settings.el_range + settings.el_resolution,
            settings.el_resolution,
        )
        self._bf_holdoff = settings.bf_holdoff_seconds
        self._last_bf_active_time: float = 0.0
        self._bf_peak_threshold = settings.bf_peak_threshold
        self._bf_min_separation_deg = settings.bf_min_separation_deg
        self._bf_max_peaks = settings.bf_max_peaks

        # Multi-peak storage (Phase 18 will consume the list)
        self.latest_peaks: list[PeakDetection] = []

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

    def process_chunk(self, chunk: np.ndarray) -> list[PeakDetection]:
        """Run beamforming on audio chunk, gated by CNN detection state.

        Args:
            chunk: (chunk_samples, num_channels) float32 array from ring buffer.

        Returns:
            List of detected peaks (empty if beamforming is inactive).
        """
        now = time.monotonic()

        # Demand-driven gate (BF-16): check CNN state
        # If no state machine, always run beamforming (backward compat)
        if self._state_machine is not None:
            from acoustic.classification.state_machine import DetectionState

            cnn_active = self._state_machine.state == DetectionState.CONFIRMED
            if cnn_active:
                self._last_bf_active_time = now
            bf_should_run = (now - self._last_bf_active_time) < self._bf_holdoff
        else:
            bf_should_run = True

        if not bf_should_run:
            self.latest_map = np.zeros(
                (len(self._az_grid), len(self._el_grid)), dtype=np.float32
            )
            self.latest_peak = None
            self.latest_peaks = []
            self._last_process_time = now
            return []

        # Transpose to (n_mics, n_samples) for beamforming
        signals = chunk.T

        # BF-11: Bandpass pre-filter
        filtered = self._bandpass.apply(signals)

        # BF-15: Real SRP-PHAT
        srp_map = srp_phat_2d(
            filtered,
            self._mic_positions,
            self._settings.sample_rate,
            self._settings.speed_of_sound,
            self._az_grid,
            self._el_grid,
            fmin=self._settings.bf_freq_min,
            fmax=self._settings.bf_freq_max,
        )
        self.latest_map = srp_map.astype(np.float32)

        # BF-14: MCRA noise floor
        noise_floor = self._mcra.update(srp_map)

        # BF-13: Multi-peak detection
        peaks = detect_multi_peak(
            srp_map,
            self._az_grid,
            self._el_grid,
            noise_floor,
            threshold_factor=self._bf_peak_threshold,
            min_separation_deg=self._bf_min_separation_deg,
            max_peaks=self._bf_max_peaks,
        )

        # BF-12: Refine each peak with parabolic interpolation
        for i, pk in enumerate(peaks):
            az_idx = int(np.argmin(np.abs(self._az_grid - pk.az_deg)))
            el_idx = int(np.argmin(np.abs(self._el_grid - pk.el_deg)))
            refined_az, refined_el = parabolic_interpolation_2d(
                srp_map, az_idx, el_idx, self._az_grid, self._el_grid,
            )
            peaks[i] = PeakDetection(
                az_deg=refined_az,
                el_deg=refined_el,
                power=pk.power,
                threshold=pk.threshold,
            )

        self.latest_peaks = peaks
        self.latest_peak = peaks[0] if peaks else None
        self._last_process_time = now
        return peaks

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
                    peaks = self.process_chunk(chunk)
                    self._process_cnn(chunk, peaks)
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

    def _process_cnn(self, chunk: np.ndarray, peaks: list[PeakDetection]) -> None:
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
        # - On peak: push immediately with best peak bearing
        # - No peak: push periodically (every _cnn_interval) with (0,0) bearing
        #   so the UI always has a fresh drone probability
        now = time.monotonic()
        if self._mono_buffer_samples >= self._cnn_segment_samples:
            segment = np.concatenate(self._mono_buffer)[-self._cnn_segment_samples:]
            if peaks:
                best = peaks[0]
                self._cnn_worker.push(segment, best.az_deg, best.el_deg)
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
        self.latest_peaks = []
        self._last_process_time = None
        self._bandpass.reset(self._settings.num_channels)
        self._mcra.reset()
        self._last_bf_active_time = 0.0
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
