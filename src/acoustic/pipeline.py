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
        self.latest_mass_center: dict | None = None
        self._srp_accumulator: np.ndarray | None = None  # temporal smoothing
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

        # Spectrum analyzer: band energies computed per chunk
        self.latest_spectrum: dict | None = None

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

        # Target location recording
        self._target_recording: bool = False
        self._target_recording_id: str | None = None
        self._target_recording_start: float | None = None
        self._target_recording_frames: list[dict] = []

    def process_chunk(self, chunk: np.ndarray) -> list[PeakDetection]:
        """Run beamforming on audio chunk, gated by CNN detection state.

        Args:
            chunk: (chunk_samples, num_channels) float32 array from ring buffer.

        Returns:
            List of detected peaks (empty if beamforming is inactive).
        """
        now = time.monotonic()

        # Demand-driven gate (BF-16): check CNN state
        # bf_always_on bypasses CNN gating — beamforming runs on every chunk
        if self._settings.bf_always_on:
            bf_should_run = True
        elif self._state_machine is not None:
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

        # Fold elevation: planar array (z=0) has perfect +el/-el symmetry.
        # Average the two halves for 3dB SNR gain, then keep only el >= 0.
        n_el = srp_map.shape[1]
        mid = n_el // 2
        # Average symmetric pairs into the upper half (el >= 0)
        for i in range(mid + 1):
            if mid + i < n_el and mid - i >= 0:
                srp_map[:, mid + i] = (srp_map[:, mid + i] + srp_map[:, mid - i]) * 0.5
        # Crop to el >= 0 only (indices mid..end)
        srp_map = srp_map[:, mid:]

        # Temporal smoothing: EMA accumulator stabilizes the peak across frames.
        # Without this, frame-to-frame noise causes the peak to jump randomly.
        alpha = 0.3  # blend factor: 0.3 = 30% new frame, 70% history
        if self._srp_accumulator is None or self._srp_accumulator.shape != srp_map.shape:
            self._srp_accumulator = srp_map.copy()
        else:
            self._srp_accumulator = alpha * srp_map + (1 - alpha) * self._srp_accumulator

        # Normalize to [0,1] using min-max on the smoothed map.
        # Then apply moderate power exponent for contrast (nu).
        smoothed = self._srp_accumulator
        smin = smoothed.min()
        smax = smoothed.max()
        if smax > smin:
            norm_map = (smoothed - smin) / (smax - smin)
            # Apply functional beamforming exponent for contrast.
            # Low nu (2-10) shows broad blobs; high nu (50+) sharpens to peaks.
            nu = self._settings.bf_nu
            fb_map = norm_map ** nu
            fb_map[fb_map < 1e-6] = 0.0
            self.latest_map = fb_map.astype(np.float32)

            # Center of mass on the thresholded map
            # Apply cos(el) correction: planar arrays (z=0) bias SRP power
            # toward high elevation (broadside) because TDOAs shrink with
            # increasing el. Weighting by cos(el) compensates.
            el_cropped = self._el_grid[len(self._el_grid) // 2:]  # el >= 0
            cos_el = np.cos(np.deg2rad(el_cropped))  # (n_el_half,)
            weighted_map = fb_map * cos_el[np.newaxis, :]

            mask = weighted_map > self._settings.bf_mass_threshold
            if mask.any():
                weights = weighted_map[mask]
                az_indices, el_indices = np.where(mask)
                # Map indices to degrees using the cropped grids
                az_vals = self._az_grid[az_indices]
                el_vals = el_cropped[el_indices] if len(el_cropped) > el_indices.max() else np.zeros_like(el_indices, dtype=float)
                total_w = weights.sum()
                com_az = float(np.sum(az_vals * weights) / total_w)
                com_el = float(np.sum(el_vals * weights) / total_w)
                # Weighted std dev as error estimation (±1σ around center)
                az_var = float(np.sum(weights * (az_vals - com_az) ** 2) / total_w)
                el_var = float(np.sum(weights * (el_vals - com_el) ** 2) / total_w)
                az_std = float(np.sqrt(az_var))
                el_std = float(np.sqrt(el_var))
                self.latest_mass_center = {
                    "az_deg": round(com_az, 1),
                    "el_deg": round(com_el, 1),
                    "az_min": round(com_az - az_std, 1),
                    "az_max": round(com_az + az_std, 1),
                    "el_min": round(com_el - el_std, 1),
                    "el_max": round(com_el + el_std, 1),
                }
            else:
                self.latest_mass_center = None
        else:
            self.latest_map = np.zeros_like(srp_map, dtype=np.float32)
            self.latest_mass_center = None

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
            # Skip live audio while file playback is active
            if getattr(self, "_playback_path", None) is not None:
                ring_buffer.read()  # drain buffer to prevent overflow
                time.sleep(0.01)
                continue
            chunk = ring_buffer.read()
            if chunk is not None:
                try:
                    # Track raw (pre-gain) mono RMS for the sound-level meter
                    if chunk.size > 0:
                        mono_raw = chunk.mean(axis=1)
                        self._latest_raw_rms = float(np.sqrt(np.mean(mono_raw.astype(np.float32) ** 2)))
                        self._compute_spectrum(mono_raw)
                    # Forward to recording manager (passive observer)
                    if self._recording_manager is not None:
                        self._recording_manager.feed_chunk(chunk)
                    # Feed raw 16ch recording if active
                    self._feed_raw_recording(chunk)
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

        # Sample targets for target location recording
        self._sample_targets()

    def _sample_targets(self) -> None:
        """Append current target snapshot if target recording is active."""
        if not self._target_recording:
            return
        targets = self.latest_targets
        if targets:
            self._target_recording_frames.append({
                "t": round(time.time(), 3),
                "targets": targets,
            })

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

    # --- Spectrum analyzer ---------------------------------------------------

    _SPECTRUM_BANDS = [
        ("Sub-bass", 20, 60),
        ("Bass", 60, 250),
        ("Low-mid", 250, 500),
        ("Mid", 500, 2000),
        ("Upper-mid", 2000, 4000),
        ("Presence", 4000, 6000),
        ("Brilliance", 6000, 20000),
    ]

    def _compute_spectrum(self, mono: np.ndarray) -> None:
        """Compute per-band energy from mono audio chunk via FFT."""
        sr = self._settings.sample_rate
        n = len(mono)
        if n < 64:
            return
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(mono.astype(np.float32) * window))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        bands = []
        for name, flo, fhi in self._SPECTRUM_BANDS:
            mask = (freqs >= flo) & (freqs < fhi)
            if mask.any():
                energy = float(np.mean(spectrum[mask] ** 2))
                db = float(10.0 * np.log10(max(energy, 1e-20)))
            else:
                db = -100.0
            bands.append({"name": name, "fmin": flo, "fmax": fhi, "db": db})
        self.latest_spectrum = {"bands": bands, "sample_rate": sr}

    # --- Raw 16-channel recording --------------------------------------------

    def start_raw_recording(self, output_dir: str = "data/raw_recordings") -> str:
        """Start recording raw 16-channel 48kHz audio to a WAV file.

        Returns the recording ID (filename stem). Auto-stops after 60s.
        """
        import uuid
        from datetime import datetime
        from pathlib import Path

        rec_dir = Path(output_dir)
        rec_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_id = f"{ts}_{uuid.uuid4().hex[:6]}"
        path = rec_dir / f"{rec_id}.wav"

        import soundfile as sf
        f = sf.SoundFile(
            str(path), mode="w",
            samplerate=self._settings.sample_rate,
            channels=self._settings.num_channels,
            format="WAV", subtype="FLOAT",
        )
        self._raw_rec_file = f
        self._raw_rec_id = rec_id
        self._raw_rec_path = path
        self._raw_rec_start = time.monotonic()
        self._raw_rec_max_seconds = 60.0
        self._raw_rec_samples = 0
        logger.info("Raw 16ch recording started: %s", path)
        return rec_id

    def stop_raw_recording(self) -> dict | None:
        """Stop the current raw recording. Returns metadata or None."""
        f = getattr(self, "_raw_rec_file", None)
        if f is None:
            return None
        f.close()
        duration = self._raw_rec_samples / self._settings.sample_rate
        info = {
            "id": self._raw_rec_id,
            "path": str(self._raw_rec_path),
            "duration_s": round(duration, 2),
            "channels": self._settings.num_channels,
            "sample_rate": self._settings.sample_rate,
        }
        logger.info("Raw 16ch recording stopped: %.1fs, %s", duration, self._raw_rec_path)
        self._raw_rec_file = None
        return info

    @property
    def raw_recording_state(self) -> dict:
        """Return current raw recording state for WebSocket/REST."""
        f = getattr(self, "_raw_rec_file", None)
        if f is None:
            return {"status": "idle"}
        elapsed = time.monotonic() - self._raw_rec_start
        remaining = max(0, self._raw_rec_max_seconds - elapsed)
        return {
            "status": "recording",
            "id": self._raw_rec_id,
            "elapsed_s": round(elapsed, 1),
            "remaining_s": round(remaining, 1),
        }

    def _feed_raw_recording(self, chunk: np.ndarray) -> None:
        """Write chunk to raw recording if active. Auto-stop at limit."""
        f = getattr(self, "_raw_rec_file", None)
        if f is None:
            return
        elapsed = time.monotonic() - self._raw_rec_start
        if elapsed >= self._raw_rec_max_seconds:
            self.stop_raw_recording()
            return
        f.write(chunk)
        self._raw_rec_samples += chunk.shape[0]

    # --- Target location recording -----------------------------------------------

    def start_target_recording(self) -> str:
        """Start recording target locations to a JSON file. Returns recording ID."""
        from datetime import datetime

        rec_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._target_recording = True
        self._target_recording_id = rec_id
        self._target_recording_start = time.time()
        self._target_recording_frames = []
        logger.info("Target recording started: %s", rec_id)
        return rec_id

    def stop_target_recording(self) -> dict | None:
        """Stop target recording and write JSON file. Returns metadata or None."""
        if not self._target_recording:
            return None
        import json
        from pathlib import Path

        self._target_recording = False
        rec_id = self._target_recording_id
        started_at = self._target_recording_start or 0
        stopped_at = time.time()
        frames = self._target_recording_frames

        out_dir = Path("data/target_recordings")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{rec_id}.json"

        payload = {
            "id": rec_id,
            "started_at": round(started_at, 3),
            "stopped_at": round(stopped_at, 3),
            "duration_s": round(stopped_at - started_at, 3),
            "total_samples": len(frames),
            "frames": frames,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Target recording saved: %s (%d samples, %.1fs)", path, len(frames), stopped_at - started_at)
        self._target_recording_frames = []
        return {
            "id": rec_id,
            "status": "saved",
            "path": str(path),
            "total_samples": len(frames),
            "duration_s": round(stopped_at - started_at, 3),
        }

    @property
    def target_recording_state(self) -> dict:
        """Return current target recording state for WebSocket/REST."""
        if self._target_recording:
            elapsed = time.time() - (self._target_recording_start or 0)
            return {
                "status": "recording",
                "id": self._target_recording_id,
                "elapsed_s": round(elapsed, 1),
                "samples": len(self._target_recording_frames),
            }
        return {"status": "idle"}

    # --- File playback --------------------------------------------------------

    def start_file_playback(self, wav_path: str) -> None:
        """Play a WAV file through the pipeline instead of live audio.

        Spawns a thread that reads the file in real-time chunks and feeds them
        through process_chunk, so the heatmap/spectrum/peaks update as if live.
        """
        self._playback_stop = False
        self._playback_path = wav_path
        self._srp_accumulator = None  # reset temporal smoothing
        self._bandpass.reset(self._settings.num_channels)
        self._playback_thread = threading.Thread(
            target=self._playback_loop, args=(wav_path,), daemon=True
        )
        self._playback_thread.start()
        logger.info("File playback started: %s", wav_path)

    def stop_file_playback(self) -> None:
        """Stop current file playback."""
        self._playback_stop = True
        t = getattr(self, "_playback_thread", None)
        if t is not None:
            t.join(timeout=3.0)
            self._playback_thread = None
        self._playback_path = None
        logger.info("File playback stopped")

    @property
    def playback_state(self) -> dict:
        """Return current playback state."""
        path = getattr(self, "_playback_path", None)
        if path is None:
            return {"status": "idle"}
        return {"status": "playing", "path": path}

    def _playback_loop(self, wav_path: str) -> None:
        """Read WAV file in real-time chunks and process through pipeline."""
        import soundfile as sf

        chunk_samples = self._settings.chunk_samples
        sr = self._settings.sample_rate
        chunk_duration = chunk_samples / sr

        try:
            with sf.SoundFile(wav_path, "r") as f:
                file_sr = f.samplerate
                file_ch = f.channels
                if file_sr != sr:
                    logger.warning("Playback SR mismatch: file=%d, pipeline=%d", file_sr, sr)

                logger.info(
                    "Playback: %dch %dHz, %d frames (%.1fs)",
                    file_ch, file_sr, f.frames, f.frames / file_sr,
                )

                while not self._playback_stop:
                    data = f.read(chunk_samples, dtype="float32")
                    if data.size == 0:
                        break  # EOF

                    # Ensure shape is (samples, num_channels)
                    if data.ndim == 1:
                        data = np.tile(data[:, np.newaxis], (1, self._settings.num_channels))
                    elif data.shape[1] != self._settings.num_channels:
                        # Pad or trim channels
                        if data.shape[1] < self._settings.num_channels:
                            pad = np.zeros(
                                (data.shape[0], self._settings.num_channels - data.shape[1]),
                                dtype=np.float32,
                            )
                            data = np.hstack([data, pad])
                        else:
                            data = data[:, :self._settings.num_channels]

                    try:
                        mono_raw = data.mean(axis=1)
                        self._latest_raw_rms = float(np.sqrt(np.mean(mono_raw.astype(np.float32) ** 2)))
                        self._compute_spectrum(mono_raw)
                        peaks = self.process_chunk(data)
                        self._process_cnn(data, peaks)
                    except Exception:
                        logger.exception("Error processing playback chunk")

                    time.sleep(chunk_duration)

        except Exception:
            logger.exception("Playback file error: %s", wav_path)
        finally:
            self._playback_path = None
            logger.info("Playback finished: %s", wav_path)

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
