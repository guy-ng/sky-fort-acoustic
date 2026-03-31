"""Background device monitor that polls for UMA-16v2 presence changes."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

from acoustic.audio.device import detect_uma16v2
from acoustic.types import DeviceInfo

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 3.0  # seconds


@dataclass
class DeviceStatus:
    """Snapshot of current device state."""

    detected: bool
    name: str | None
    scanning: bool


class DeviceMonitor:
    """Polls for UMA-16v2 and notifies async subscribers on state changes.

    Runs a daemon thread that calls detect_uma16v2() periodically.
    When the detected/not-detected state changes, all registered
    asyncio.Queue subscribers receive a DeviceStatus message.
    """

    def __init__(self, poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._device_info: DeviceInfo | None = None
        self._detected = False
        self._subscribers: list[asyncio.Queue[DeviceStatus]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

    @property
    def detected(self) -> bool:
        return self._detected

    def subscribe(self) -> asyncio.Queue[DeviceStatus]:
        """Create a new subscriber queue. Caller must call unsubscribe() when done."""
        q: asyncio.Queue[DeviceStatus] = asyncio.Queue(maxsize=16)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[DeviceStatus]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, status: DeviceStatus) -> None:
        """Push status to all subscribers (thread-safe, non-blocking)."""
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(q.put_nowait, status)

    def _poll(self) -> None:
        while self._running:
            try:
                info = detect_uma16v2()
            except Exception:
                logger.exception("Device detection error")
                info = None

            new_detected = info is not None
            if new_detected != self._detected or (info and self._device_info and info.name != self._device_info.name):
                self._device_info = info
                self._detected = new_detected
                state = "connected" if new_detected else "disconnected"
                logger.info("Device %s: %s", state, info.name if info else "none")
                self._broadcast(DeviceStatus(
                    detected=new_detected,
                    name=info.name if info else None,
                    scanning=not new_detected,
                ))

            time.sleep(self._poll_interval)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._loop = loop
        self._running = True
        # Initial detection
        try:
            info = detect_uma16v2()
        except Exception:
            info = None
        self._device_info = info
        self._detected = info is not None
        self._thread = threading.Thread(target=self._poll, daemon=True, name="device-monitor")
        self._thread.start()
        logger.info("Device monitor started (initial: %s)", "detected" if self._detected else "not detected")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def current_status(self) -> DeviceStatus:
        return DeviceStatus(
            detected=self._detected,
            name=self._device_info.name if self._device_info else None,
            scanning=not self._detected,
        )
