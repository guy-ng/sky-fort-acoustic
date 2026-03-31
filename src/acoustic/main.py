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
from acoustic.audio.capture import AudioCapture, AudioRingBuffer
from acoustic.audio.device import detect_uma16v2
from acoustic.audio.monitor import DeviceMonitor
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


def _create_hardware_capture(settings: AcousticSettings, device_index: int | str) -> AudioCapture:
    """Create and start a new AudioCapture for the given device."""
    capture = AudioCapture(
        device=device_index,
        fs=settings.sample_rate,
        channels=settings.num_channels,
        chunk_samples=settings.chunk_samples,
        ring_chunks=settings.ring_chunks,
    )
    capture.start()
    return capture


async def _device_lifecycle_task(app: FastAPI) -> None:
    """Watch for device disconnect/reconnect and rebuild the audio pipeline.

    Subscribes to DeviceMonitor events. On disconnect, safely tears down
    the capture and clears pipeline state. On reconnect, creates a new
    AudioCapture and restarts the pipeline with the new ring buffer.

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
                app.state.pipeline.clear_state()
                logger.info("Audio capture stopped, pipeline state cleared")

            else:
                # --- Device reconnected ---
                device_info = detect_uma16v2()
                if device_info is None:
                    logger.warning("DeviceMonitor reported connected but detect_uma16v2() returned None")
                    continue

                logger.info("Device reconnected: %s (index=%s)", device_info.name, device_info.index)

                # Stop old capture if it somehow still exists
                old_capture = app.state.capture
                if old_capture is not None:
                    old_capture.stop()

                # Create new capture and restart pipeline
                capture = _create_hardware_capture(settings, device_info.index)
                app.state.capture = capture
                app.state.device_info = device_info
                app.state.pipeline.restart(capture.ring)
                logger.info("Audio pipeline rebuilt with new device")

    except asyncio.CancelledError:
        pass
    finally:
        monitor.unsubscribe(queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage audio capture and beamforming pipeline lifecycle."""
    settings = AcousticSettings()
    device_info = detect_uma16v2()

    # D-04: Auto-switch to simulated mode if no hardware detected
    if device_info is None and settings.audio_source != "simulated":
        logger.warning(
            "UMA-16v2 not detected, auto-switching to simulated audio source (D-04)"
        )
        settings.audio_source = "simulated"

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
    else:
        logger.info("Starting with hardware audio capture (device=%s)", device_info.index)
        capture = _create_hardware_capture(settings, device_info.index)

    pipeline = BeamformingPipeline(settings)
    pipeline.start(capture.ring)

    # Start device monitor for live hot-plug detection
    device_monitor = DeviceMonitor(poll_interval=3.0)
    device_monitor.start(asyncio.get_running_loop())

    # Store on app.state for endpoint access
    app.state.settings = settings
    app.state.device_info = device_info
    app.state.capture = capture
    app.state.pipeline = pipeline
    app.state.device_monitor = device_monitor

    # Start background lifecycle task for hot-plug recovery
    lifecycle_task = asyncio.create_task(_device_lifecycle_task(app))

    logger.info("Acoustic service started (pipeline running)")
    yield

    lifecycle_task.cancel()
    try:
        await lifecycle_task
    except asyncio.CancelledError:
        pass

    pipeline.stop()
    if app.state.capture is not None:
        app.state.capture.stop()
    device_monitor.stop()
    logger.info("Acoustic service stopped")


app = FastAPI(title="Sky Fort Acoustic Service", lifespan=lifespan)
app.include_router(api_router)
app.include_router(ws_router)


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
