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
from acoustic.classification.efficientat.window_contract import (
    EFFICIENTAT_WINDOW_SECONDS,
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
    # Absolute mic-calibration multiplier — applied inside the preprocessor,
    # never stacked with anything else. Default tuned for UMA-16v2.
    gain: float = 500.0
    # window_seconds is derived from model type to match training (NOT user-tunable).
    window_seconds: float = 0.5
    interval_seconds: float = 0.2
    log: deque = field(default_factory=lambda: deque(maxlen=MAX_DETECTION_LOG))


def _training_window_seconds(model_type: str) -> float:
    """Return the audio window length the given model was TRAINED on.

    The window pushed to the classifier at runtime must always match training
    or the mel-spectrogram time axis will not align with the model input. These
    values are sourced from the actual training configs:
        - research_cnn: MelConfig.segment_seconds = 0.5 (16 kHz mel)
        - efficientat: EfficientATMelConfig — input_dim_t * hop_size / sample_rate
                       = 100 * 320 / 32000 = 1.0 s
    """
    mt = (model_type or "").lower()
    if "efficientat" in mt or "mn10" in mt or "mn05" in mt:
        return EFFICIENTAT_WINDOW_SECONDS
    # default to research_cnn training window — intentional, different window
    return 0.5


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
        self._latest_raw_rms: float | None = None  # pre-gain RMS for level meter
        # CNN cadence: every `_cnn_interval` seconds, push the last
        # `_cnn_segment_samples` samples to the classifier. The WINDOW length
        # is set per session by start_detection_session() to match the active
        # model's training window. The default below (research_cnn / 0.5 s) is
        # only a placeholder until a session starts.
        self._cnn_segment_samples: int = int(settings.sample_rate * _training_window_seconds("research_cnn"))
        self._default_segment_samples: int = self._cnn_segment_samples
        self._last_cnn_push: float = 0.0
        self._cnn_interval: float = settings.cnn_interval_seconds
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
        # VIZ-02: Functional beamforming for sidelobe suppression (D-01, D-07)
        max_val = srp_map.max()
        if max_val > 0:
            fb_map = (srp_map / max_val) ** self._settings.bf_nu
            fb_map[fb_map < 1e-6] = 0.0
            self.latest_map = fb_map.astype(np.float32)
        else:
            self.latest_map = np.zeros_like(srp_map, dtype=np.float32)

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
                    # Track raw (pre-gain) mono RMS for the sound-level meter
                    if chunk.size > 0:
                        mono_raw = chunk.mean(axis=1)
                        self._latest_raw_rms = float(np.sqrt(np.mean(mono_raw.astype(np.float32) ** 2)))
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
        interval_seconds: float | None = None,
    ) -> None:
        """Start a user-initiated detection session.

        The CNN audio window length is derived from `model_type` so it always
        matches what the model was trained on (research_cnn → 0.5 s,
        efficientat → 1.0 s). Only the inference INTERVAL is user-tunable —
        every `interval_seconds` the pipeline classifies the most recent
        training-window of audio.
        """
        # Window length is dictated by the model — never user-overridable.
        win = _training_window_seconds(model_type)
        ivl = float(interval_seconds) if interval_seconds is not None else self._settings.cnn_interval_seconds
        if ivl <= 0:
            raise ValueError("interval_seconds must be > 0")

        with self._detection_lock:
            self._detection_session = DetectionSession(
                model_path=model_path,
                confidence=confidence,
                time_frame=time_frame,
                positive_detections=positive_detections,
                gain=gain,
                window_seconds=win,
                interval_seconds=ivl,
            )

        self._cnn_segment_samples = int(self._settings.sample_rate * win)
        self._cnn_interval = ivl

        # Reset mono buffer for new segment size
        self._mono_buffer.clear()
        self._mono_buffer_samples = 0
        self._last_cnn_push = 0.0

        # Reconfigure state machine thresholds
        if self._state_machine is not None:
            self._state_machine._enter = confidence
            self._state_machine._exit = max(0.1, confidence - 0.4)
            self._state_machine._confirm = positive_detections
            self._state_machine.reset()
        logger.info(
            "Detection session started: model=%s type=%s window=%.2fs (training-locked) interval=%.2fs confidence=%.0f%% gain=%.1f",
            model_path, model_type, win, ivl, confidence * 100, gain,
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

        # Always accumulate mono audio so we have a rolling buffer ready.
        # NOTE: gain is NOT applied here. The session `gain` is the single
        # absolute mic-calibration multiplier and lives inside the preprocessor
        # (RawAudioPreprocessor.input_gain) so it survives a hot model swap and
        # never stacks with anything. See pipeline_routes.start_detection().
        mono = chunk.mean(axis=1).astype(np.float32)
        self._mono_buffer.append(mono)
        self._mono_buffer_samples += len(mono)

        # Trim buffer to at most 2x the segment length to bound memory
        max_samples = self._cnn_segment_samples * 2
        while self._mono_buffer_samples > max_samples:
            dropped = self._mono_buffer.pop(0)
            self._mono_buffer_samples -= len(dropped)

        # Push to CNN at a fixed cadence: every `_cnn_interval` seconds, send
        # the most recent `_cnn_segment_samples` of audio. If a peak was just
        # detected we attach its bearing, otherwise (0,0) — bearing is only a
        # hint to the tracker, not a gate on inference.
        now = time.monotonic()
        if (
            self._mono_buffer_samples >= self._cnn_segment_samples
            and now - self._last_cnn_push >= self._cnn_interval
        ):
            segment = np.concatenate(self._mono_buffer)[-self._cnn_segment_samples:]
            if peaks:
                best = peaks[0]
                self._cnn_worker.push(segment, best.az_deg, best.el_deg)
            else:
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
                if self.latest_peaks:
                    self._tracker.update_multi(
                        self.latest_peaks,
                        confidence=result.drone_probability,
                    )
                else:
                    # Fallback: no beamforming peaks available, use CNN result bearing
                    self._tracker.update(
                        result.az_deg, result.el_deg, result.drone_probability
                    )

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
    def latest_audio_level_db(self) -> float | None:
        """Return raw (pre-gain) RMS level in dBFS of the most recent audio chunk.

        Updated by `_process_cnn` on every chunk. Returns None before the first
        chunk has been processed.
        """
        if self._latest_raw_rms is None:
            return None
        return float(20.0 * np.log10(max(self._latest_raw_rms, 1e-10)))

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
