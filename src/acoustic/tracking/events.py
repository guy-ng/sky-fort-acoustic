"""WebSocket event broadcaster for target tracking events.

Uses asyncio.Queue fanout pattern (no ZeroMQ dependency).
Thread-safe for calls from the pipeline processing thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from acoustic.tracking.schema import TargetEvent

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Fan-out broadcaster that delivers TargetEvent to all WebSocket subscriber queues.

    Thread-safe: broadcast() can be called from any thread.
    Uses loop.call_soon_threadsafe() to properly wake up async consumers
    when called from non-asyncio threads (e.g., the pipeline processing thread).
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new subscriber queue.

        Captures the running event loop on first subscribe so broadcast()
        can safely deliver events from non-async threads.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(q)
            if self._loop is None:
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            self._subscribers.discard(queue)

    def broadcast(self, event: TargetEvent) -> None:
        """Send event to all subscribers. Non-blocking; drops if queue full.

        Thread-safe: uses call_soon_threadsafe when called from a non-asyncio thread.
        """
        data = event.model_dump()
        with self._lock:
            subscribers = list(self._subscribers)
            loop = self._loop

        if loop is not None and loop.is_running():
            # Called from a non-event-loop thread (pipeline thread) --
            # schedule delivery on the event loop to properly wake async consumers
            try:
                loop.call_soon_threadsafe(self._deliver, data, subscribers)
                return
            except RuntimeError:
                # Loop closed -- fall through to direct delivery
                pass

        # Direct delivery (same thread as event loop, or loop not available)
        self._deliver(data, subscribers)

    def _deliver(self, data: dict, subscribers: list[asyncio.Queue]) -> None:
        """Put data into all subscriber queues (non-blocking)."""
        for q in subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event for subscriber")
