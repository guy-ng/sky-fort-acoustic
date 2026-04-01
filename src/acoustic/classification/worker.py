"""Background CNN inference worker thread.

Runs inference in a daemon thread so it never blocks the beamforming loop.
Uses a single-slot queue with drop semantics (latest audio only).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from acoustic.classification.protocols import Classifier, Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of a single CNN inference pass."""

    drone_probability: float
    timestamp: float
    az_deg: float
    el_deg: float


class CNNWorker:
    """Background thread that runs CNN drone classification on audio segments.

    Push mono audio via push(); poll results via get_latest().
    Queue is maxsize=1 with drop semantics -- only the most recent audio is kept.
    """

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        classifier: Classifier | None = None,
        *,
        fs_in: int = 48000,
        silence_threshold: float = 0.001,
    ) -> None:
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._fs_in = fs_in
        self._silence_threshold = silence_threshold
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest: ClassificationResult | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background inference thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cnn-worker")
        self._thread.start()
        logger.info("CNN worker thread started")

    def stop(self) -> None:
        """Stop the background inference thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("CNN worker thread stopped")

    def push(self, mono_audio: np.ndarray, az_deg: float, el_deg: float) -> None:
        """Submit audio for classification (non-blocking, replace if busy).

        Args:
            mono_audio: 1-D float32 mono audio at self._fs_in sample rate.
            az_deg: Azimuth of the detected peak.
            el_deg: Elevation of the detected peak.
        """
        item = (mono_audio, az_deg, el_deg)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drain stale item and replace with latest audio so the worker
            # always processes the most recent segment after current inference.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def get_latest(self) -> ClassificationResult | None:
        """Return the most recent classification result, or None."""
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        """Background thread target: consume audio from queue, run inference."""
        while not self._stop_event.is_set():
            try:
                mono_audio, az_deg, el_deg = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()

                # Energy gate
                rms = np.sqrt(np.mean(mono_audio ** 2))
                if rms < self._silence_threshold:
                    result = ClassificationResult(
                        drone_probability=0.0,
                        timestamp=time.monotonic(),
                        az_deg=az_deg,
                        el_deg=el_deg,
                    )
                    with self._lock:
                        self._latest = result
                    logger.debug("CNN: silence detected, reporting prob=0.0")
                    continue

                # Preprocess (skip if no preprocessor -- dormant until injected)
                if self._preprocessor is None:
                    continue
                features = self._preprocessor.process(mono_audio, self._fs_in)

                # Classify (skip if no classifier -- dormant until Phase 7)
                if self._classifier is None:
                    continue
                prob = self._classifier.predict(features)
                elapsed = time.monotonic() - t0

                result = ClassificationResult(
                    drone_probability=prob,
                    timestamp=time.monotonic(),
                    az_deg=az_deg,
                    el_deg=el_deg,
                )
                with self._lock:
                    self._latest = result

                logger.debug("CNN inference: prob=%.3f in %.1fms", prob, elapsed * 1000)
            except Exception:
                logger.exception("Error in CNN inference")
