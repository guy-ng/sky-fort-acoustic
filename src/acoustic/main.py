"""FastAPI application for the Sky Fort Acoustic Service."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from acoustic.api.routes import router as api_router
from acoustic.api.static import mount_static
from acoustic.api.websocket import router as ws_router
import sounddevice as sd

from acoustic.audio.capture import AudioCapture, AudioRingBuffer
from acoustic.audio.device import detect_uma16v2
from acoustic.audio.monitor import DeviceMonitor, DeviceStatus
from acoustic.audio.simulator import SimulatedAudioSource
from acoustic.config import AcousticSettings
from acoustic.pipeline import BeamformingPipeline

logger = logging.getLogger(__name__)


class SimulatedProducer:
    """Feeds simulated audio chunks into a ring buffer in a background thread.

    Simulates real-time audio capture when no hardware is available.
    """

    def __init__(
        self,
        source: SimulatedAudioSource,
        ring: AudioRingBuffer,
        chunk_seconds: float,
    ) -> None:
        self._source = source
        self._ring = ring
        self._chunk_seconds = chunk_seconds
        self._running = False
        self._thread: threading.Thread | None = None
        self._az_deg = 0.0
        self._last_write_time: float | None = None

    def _run(self) -> None:
        """Generate and write simulated chunks at real-time rate."""
        while self._running:
            chunk = self._source.get_chunk(source_az_deg=self._az_deg)
            self._ring.write(chunk)
            self._last_write_time = time.monotonic()
            # Slowly vary azimuth for visual interest
            self._az_deg = (self._az_deg + 1.0) % 360.0 - 180.0
            time.sleep(self._chunk_seconds)

    def start(self) -> None:
        """Start the producer thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the producer thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class _CaptureShim:
    """Shim that wraps a ring buffer and producer to match AudioCapture interface.

    Used in simulated mode where we don't use sounddevice.InputStream.
    """

    def __init__(self, ring: AudioRingBuffer, producer: SimulatedProducer) -> None:
        self._ring = ring
        self._producer = producer

    @property
    def ring(self) -> AudioRingBuffer:
        return self._ring

    @property
    def last_frame_time(self) -> float | None:
        return self._producer._last_write_time

    def start(self) -> None:
        self._producer.start()

    def stop(self) -> None:
        self._producer.stop()


def _create_hardware_capture(
    settings: AcousticSettings,
    device_index: int | str,
    monitor: DeviceMonitor | None = None,
) -> AudioCapture:
    """Create and start a new AudioCapture for the given device."""
    capture = AudioCapture(
        device=device_index,
        fs=settings.sample_rate,
        channels=settings.num_channels,
        chunk_samples=settings.chunk_samples,
        ring_chunks=settings.ring_chunks,
        on_stream_finished=monitor.notify_stream_abort if monitor else None,
    )
    if monitor:
        monitor.set_frame_time_getter(lambda: capture.last_frame_time)
    capture.start()
    return capture


RECONNECT_VERIFY_TIMEOUT = 2.0  # seconds to wait for first audio frame
RECONNECT_RETRY_INTERVAL = 3.0  # seconds between reconnect attempts
INITIAL_SCAN_INTERVAL = 3.0  # seconds between device scans at startup


async def _device_lifecycle_task(app: FastAPI) -> None:
    """Watch for device disconnect/reconnect and rebuild the audio pipeline.

    Subscribes to DeviceMonitor events. On disconnect, safely tears down
    the capture and clears pipeline state. On reconnect, creates a new
    AudioCapture, verifies frames arrive, and restarts the pipeline.

    Only acts when the audio source is hardware (not simulated).
    """
    monitor: DeviceMonitor = app.state.device_monitor
    settings: AcousticSettings = app.state.settings
    queue = monitor.subscribe()

    try:
        while True:
            status = await queue.get()

            # Skip lifecycle management in simulated mode
            if settings.audio_source == "simulated":
                continue

            if not status.detected:
                # --- Device disconnected ---
                logger.warning("Device lost — tearing down audio capture")
                old_capture = app.state.capture
                if old_capture is not None:
                    old_capture.stop()

                app.state.capture = None
                app.state.device_info = None
                monitor.set_frame_time_getter(None)
                app.state.pipeline.clear_state()
                logger.info("Audio capture stopped, scanning for device...")

                # Actively retry until we get a working capture
                await _reconnect_loop(app, settings, monitor)

    except asyncio.CancelledError:
        pass
    finally:
        monitor.unsubscribe(queue)


async def _initial_scan_task(app: FastAPI) -> None:
    """Scan for device when none was found at startup, then hand off to lifecycle task."""
    settings: AcousticSettings = app.state.settings
    monitor: DeviceMonitor = app.state.device_monitor

    try:
        while True:
            await asyncio.sleep(INITIAL_SCAN_INTERVAL)

            _reset_portaudio()
            device_info = detect_uma16v2()
            if device_info is None:
                continue

            logger.info("Device found during scan: %s (index=%s)", device_info.name, device_info.index)

            try:
                capture = _create_hardware_capture(settings, device_info.index, monitor)
            except Exception as exc:
                logger.warning("Failed to create capture: %s — will retry", exc)
                continue

            await asyncio.sleep(RECONNECT_VERIFY_TIMEOUT)

            if capture.last_frame_time is not None:
                app.state.capture = capture
                app.state.device_info = device_info
                app.state.pipeline.start(capture.ring)
                monitor._detected = True
                monitor._device_info = device_info
                monitor._stall_disconnected = False
                monitor._stream_aborted.clear()
                monitor._broadcast(DeviceStatus(
                    detected=True,
                    name=device_info.name,
                    scanning=False,
                ))
                logger.info("Device connected — pipeline started")
                # Hand off to regular lifecycle task
                await _device_lifecycle_task(app)
                return
            else:
                logger.warning("Capture created but no audio frames — retrying")
                capture.stop()
                monitor.set_frame_time_getter(None)

    except asyncio.CancelledError:
        pass


def _reset_portaudio() -> None:
    """Re-initialize PortAudio to clear stale CoreAudio device cache."""
    try:
        sd._terminate()
        sd._initialize()
        logger.debug("PortAudio re-initialized")
    except Exception as exc:
        logger.warning("PortAudio reset failed: %s", exc)


async def _reconnect_loop(
    app: FastAPI,
    settings: AcousticSettings,
    monitor: DeviceMonitor,
) -> None:
    """Keep trying to create a working audio capture until successful."""
    while True:
        await asyncio.sleep(RECONNECT_RETRY_INTERVAL)

        # Reset PortAudio to clear stale CoreAudio state
        _reset_portaudio()

        device_info = detect_uma16v2()
        if device_info is None:
            continue

        logger.info("Device found: %s (index=%s) — attempting capture", device_info.name, device_info.index)

        try:
            capture = _create_hardware_capture(settings, device_info.index, monitor)
        except Exception as exc:
            logger.warning("Failed to create capture: %s — will retry", exc)
            continue

        # Wait for frames to verify the capture actually works
        await asyncio.sleep(RECONNECT_VERIFY_TIMEOUT)

        if capture.last_frame_time is not None:
            # Capture is producing frames — accept the reconnect
            app.state.capture = capture
            app.state.device_info = device_info
            app.state.pipeline.restart(capture.ring)
            monitor._detected = True
            monitor._device_info = device_info
            monitor._stall_disconnected = False
            monitor._stream_aborted.clear()
            monitor._broadcast(DeviceStatus(
                detected=True,
                name=device_info.name,
                scanning=False,
            ))
            logger.info("Audio pipeline rebuilt — device reconnected successfully")
            return
        else:
            # Capture failed to produce frames — tear it down and retry
            logger.warning("Capture created but no audio frames received — retrying")
            capture.stop()
            monitor.set_frame_time_getter(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage audio capture and beamforming pipeline lifecycle."""
    settings = AcousticSettings()
    device_info = detect_uma16v2()

    # Start device monitor early so capture can wire its callbacks
    device_monitor = DeviceMonitor(poll_interval=3.0)
    device_monitor.start(asyncio.get_running_loop())

    # Initialize CNN classification pipeline (optional -- graceful degradation)
    cnn_worker = None
    state_machine = None
    tracker = None
    broadcaster = None
    try:
        import os

        import torch

        from acoustic.classification.aggregation import WeightedAggregator
        from acoustic.classification.preprocessing import (
            RawAudioPreprocessor,
            ResearchPreprocessor,
        )
        from acoustic.classification.research_cnn import ResearchClassifier, ResearchCNN
        from acoustic.classification.state_machine import DetectionStateMachine
        from acoustic.classification.worker import CNNWorker
        from acoustic.tracking.events import EventBroadcaster
        from acoustic.tracking.tracker import TargetTracker

        broadcaster = EventBroadcaster()
        # Pick preprocessor by model type. EfficientAT does its own mel/STFT
        # internally so it needs raw waveforms (with mic-calibration gain
        # applied); ResearchCNN expects pre-computed mel-spectrograms.
        mt = (settings.cnn_model_type or "").lower()
        if "efficientat" in mt or "mn10" in mt or "mn05" in mt:
            preprocessor = RawAudioPreprocessor(
                input_gain=settings.cnn_input_gain,
                rms_normalize_target=settings.cnn_rms_normalize_target,
            )
        else:
            preprocessor = ResearchPreprocessor()

        # Classifier factory: ensemble detection first, then single-model fallback
        classifier = None
        ensemble_active = False

        # Ensemble factory (D-06): detect ensemble config file
        if settings.ensemble_config_path and os.path.isfile(settings.ensemble_config_path):
            try:
                from acoustic.classification.ensemble import (
                    EnsembleClassifier,
                    EnsembleConfig,
                    load_model as load_ensemble_model,
                )

                config = EnsembleConfig.from_file(settings.ensemble_config_path)
                if len(config.models) > 1:
                    classifiers_list = []
                    weights_list = []
                    for entry in config.models:
                        clf = load_ensemble_model(entry.type, entry.path)
                        classifiers_list.append(clf)
                        weights_list.append(entry.weight)
                    classifier = EnsembleClassifier(
                        classifiers_list, weights_list, live_mode=True
                    )
                    ensemble_active = True
                    logger.info(
                        "Loaded ensemble with %d models from %s",
                        len(classifiers_list),
                        settings.ensemble_config_path,
                    )
                elif len(config.models) == 1:
                    # Single model in ensemble config -- use it directly
                    entry = config.models[0]
                    classifier = load_ensemble_model(entry.type, entry.path)
                    logger.info(
                        "Ensemble config has 1 model, using single-model mode: %s",
                        entry.path,
                    )
            except Exception:
                logger.exception(
                    "Failed to load ensemble from %s -- falling back to single model",
                    settings.ensemble_config_path,
                )

        # Single-model fallback (D-06): load model if file exists, else dormant
        if classifier is None and os.path.isfile(settings.cnn_model_path):
            try:
                # Import efficientat package to trigger register_model side effect
                import acoustic.classification.efficientat  # noqa: F401
                from acoustic.classification.ensemble import load_model

                classifier = load_model(settings.cnn_model_type, settings.cnn_model_path)
                logger.info("Loaded %s model from %s", settings.cnn_model_type, settings.cnn_model_path)
            except Exception:
                logger.exception("Failed to load CNN model -- running without classifier")
        elif classifier is None:
            logger.warning(
                "CNN model not found at %s -- running in dormant mode",
                settings.cnn_model_path,
            )

        aggregator = WeightedAggregator(
            w_max=settings.cnn_agg_w_max,
            w_mean=settings.cnn_agg_w_mean,
        )

        cnn_worker = CNNWorker(
            preprocessor=preprocessor,
            classifier=classifier,
            aggregator=aggregator,
            fs_in=settings.sample_rate,
            silence_threshold=settings.cnn_silence_threshold,
        )
        state_machine = DetectionStateMachine(
            enter_threshold=settings.cnn_enter_threshold,
            exit_threshold=settings.cnn_exit_threshold,
            confirm_hits=settings.cnn_confirm_hits,
        )
        tracker = TargetTracker(ttl=settings.cnn_target_ttl, broadcaster=broadcaster)
        cnn_worker.start()
        classifier_mode = "ensemble" if ensemble_active else ("active" if classifier is not None else "dormant")
        logger.info(
            "CNN worker started (classifier=%s, aggregator=WeightedAggregator(w_max=%.2f, w_mean=%.2f))",
            classifier_mode,
            settings.cnn_agg_w_max,
            settings.cnn_agg_w_mean,
        )
    except Exception:
        logger.exception("Failed to initialize CNN worker — running without it")

    # Recording manager for field data collection (Phase 10)
    from acoustic.recording.config import RecordingConfig
    from acoustic.recording.manager import RecordingManager

    recording_config = RecordingConfig()
    recording_manager = RecordingManager(config=recording_config)

    pipeline = BeamformingPipeline(
        settings,
        cnn_worker=cnn_worker,
        state_machine=state_machine,
        tracker=tracker,
        recording_manager=recording_manager,
    )

    if settings.audio_source == "simulated":
        logger.info("Starting in simulated audio mode")
        sim = SimulatedAudioSource(settings)
        ring = AudioRingBuffer(
            num_chunks=settings.ring_chunks,
            chunk_samples=settings.chunk_samples,
            num_channels=settings.num_channels,
        )
        producer = SimulatedProducer(sim, ring, settings.chunk_seconds)
        capture = _CaptureShim(ring, producer)
        capture.start()
        pipeline.start(capture.ring)
    elif device_info is not None:
        logger.info("Starting with hardware audio capture (device=%s)", device_info.index)
        capture = _create_hardware_capture(settings, device_info.index, device_monitor)
        pipeline.start(capture.ring)
    else:
        # No device at startup — start with empty pipeline, scan for device
        logger.warning("UMA-16v2 not detected at startup — scanning for device...")
        capture = None

    # Store on app.state for endpoint access
    app.state.settings = settings
    app.state.device_info = device_info
    app.state.capture = capture
    app.state.pipeline = pipeline
    app.state.device_monitor = device_monitor
    app.state.event_broadcaster = broadcaster
    app.state.tracker = tracker
    app.state.recording_manager = recording_manager

    # Training manager for REST endpoints (Phase 9)
    from acoustic.classification.config import MelConfig
    from acoustic.training.config import TrainingConfig
    from acoustic.training.manager import TrainingManager

    training_manager = TrainingManager(config=TrainingConfig(), mel_config=MelConfig())
    app.state.training_manager = training_manager

    # Start background lifecycle task for hot-plug recovery and initial scan
    if capture is None and settings.audio_source != "simulated":
        # No device at startup — start scanning immediately
        lifecycle_task = asyncio.create_task(_initial_scan_task(app))
    else:
        lifecycle_task = asyncio.create_task(_device_lifecycle_task(app))

    logger.info("Acoustic service started%s", " (scanning for device)" if capture is None else " (pipeline running)")
    yield

    lifecycle_task.cancel()
    try:
        await lifecycle_task
    except asyncio.CancelledError:
        pass

    pipeline.stop()
    if cnn_worker is not None:
        cnn_worker.stop()
    if app.state.capture is not None:
        app.state.capture.stop()
    device_monitor.stop()
    logger.info("Acoustic service stopped")


app = FastAPI(title="Sky Fort Acoustic Service", lifespan=lifespan)
app.include_router(api_router)
app.include_router(ws_router)

# Phase 9: Training, evaluation, and model listing routes
from acoustic.api.eval_routes import router as eval_router
from acoustic.api.model_routes import router as model_router
from acoustic.api.training_routes import router as training_router

app.include_router(training_router)
app.include_router(eval_router)
app.include_router(model_router)

# Phase 12: Pipeline control (model activation)
from acoustic.api.pipeline_routes import router as pipeline_router

app.include_router(pipeline_router)

# Phase 10: Recording routes
from acoustic.api.recording_routes import router as recording_router

app.include_router(recording_router)

# Test pipeline routes (DADS sample preview in Tools page)
from acoustic.api.test_pipeline_routes import router as test_pipeline_router

app.include_router(test_pipeline_router)


@app.get("/health")
async def health():
    """Return service health status with live pipeline state."""
    pipeline = app.state.pipeline
    capture = app.state.capture
    monitor = app.state.device_monitor

    return {
        "status": "ok" if pipeline.running else "degraded",
        "device_detected": monitor.detected,
        "device_name": monitor.device_info.name if monitor.device_info else None,
        "pipeline_running": pipeline.running,
        "overflow_count": capture.ring.overflow_count if capture is not None else 0,
        "last_frame_time": capture.last_frame_time if capture is not None else None,
    }


# SPA static mount MUST be last -- catch-all route would shadow API routes
mount_static(app)
