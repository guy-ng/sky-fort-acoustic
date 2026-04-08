"""Shared mutable state between the main loop and the HTTP handler thread.

The HTTP server (LocalhostJSONServer) runs in a background thread and must
read a consistent view of the pipeline's runtime state (model load status,
audio stream health, last inference / detection timestamps, LED state, log
file path). This module exposes a small thread-safe dataclass that both the
main loop and the handler share via the ``snapshot``/``update`` API.

The lock is stored in a private attribute so ``dataclasses.asdict`` can skip
it when producing JSON-friendly snapshots.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RuntimeState:
    model_loaded: bool = False
    audio_stream_alive: bool = False
    last_inference_time: Optional[float] = None
    last_detection_time: Optional[float] = None
    led_state: str = "off"
    log_file_path: str = ""
    active_model_path: str = ""

    def __post_init__(self) -> None:
        # Use object.__setattr__ so dataclass(frozen=False) still treats
        # the field as an instance attribute outside the declared fields.
        object.__setattr__(self, "_lock", threading.Lock())

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly dict snapshot of the current state."""
        with self._lock:  # type: ignore[attr-defined]
            d = asdict(self)
        # asdict() should not include _lock because it's not a dataclass field,
        # but strip it defensively in case of subclass shenanigans.
        d.pop("_lock", None)
        return d

    def update(self, **kwargs: Any) -> None:
        """Thread-safely update one or more fields.

        Unknown keys and private attributes are silently ignored so callers
        can pass partial payloads without defensive filtering.
        """
        with self._lock:  # type: ignore[attr-defined]
            for k, v in kwargs.items():
                if k.startswith("_"):
                    continue
                if hasattr(self, k):
                    setattr(self, k, v)
