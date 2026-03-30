"""FastAPI application for the Sky Fort Acoustic Service."""

from __future__ import annotations

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
        capture = AudioCapture(
            device=device_info.index,
            fs=settings.sample_rate,
            channels=settings.num_channels,
            chunk_samples=settings.chunk_samples,
            ring_chunks=settings.ring_chunks,
        )
        capture.start()

    pipeline = BeamformingPipeline(settings)
    pipeline.start(capture.ring)

    # Store on app.state for endpoint access
    app.state.settings = settings
    app.state.device_info = device_info
    app.state.capture = capture
    app.state.pipeline = pipeline

    logger.info("Acoustic service started (pipeline running)")
    yield

    pipeline.stop()
    capture.stop()
    logger.info("Acoustic service stopped")


app = FastAPI(title="Sky Fort Acoustic Service", lifespan=lifespan)
app.include_router(api_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    """Return service health status with live pipeline state."""
    pipeline = app.state.pipeline
    capture = app.state.capture

    return {
        "status": "ok" if pipeline.running else "degraded",
        "device_detected": app.state.device_info is not None,
        "pipeline_running": pipeline.running,
        "overflow_count": capture.ring.overflow_count,
        "last_frame_time": capture.last_frame_time,
    }


# SPA static mount MUST be last -- catch-all route would shadow API routes
mount_static(app)
