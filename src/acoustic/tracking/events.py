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
    Subscribers are asyncio.Queue instances consumed by WebSocket handlers.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            self._subscribers.discard(queue)

    def broadcast(self, event: TargetEvent) -> None:
        """Send event to all subscribers. Non-blocking; drops if queue full."""
        data = event.model_dump()
        with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    logger.warning("Event queue full, dropping event for subscriber")
