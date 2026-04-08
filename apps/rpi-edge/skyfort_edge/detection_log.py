"""D-20/D-21/D-22: Always-on rotating JSONL detection log.

CANNOT be disabled via config (D-21/D-22). `DetectionLogConfig` intentionally
has no `enabled` field. The general app log level (D-23) controls journald
verbosity, NOT this log — a dedicated logger with `propagate=False` keeps the
two streams isolated (T-21-14 mitigation).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from skyfort_edge.hysteresis import StateEvent

_DETECTION_LOGGER_NAME = "skyfort_edge.detection"


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {"message": record.getMessage()}
        return json.dumps(payload, separators=(",", ":"))


class DetectionLogger:
    """Dedicated always-on rotating JSONL logger for detection events."""

    def __init__(self, cfg) -> None:
        self._path = Path(cfg.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(_DETECTION_LOGGER_NAME)
        self._logger.setLevel(logging.INFO)
        # D-21/D-23: isolate from general logging so silencing the root logger
        # never silences this forensic record.
        self._logger.propagate = False

        # Remove any stale handlers (test re-entry / repeated instantiation)
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        handler = RotatingFileHandler(
            filename=str(self._path),
            maxBytes=cfg.rotate_max_bytes,
            backupCount=cfg.rotate_backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(_JsonLineFormatter())
        self._logger.addHandler(handler)
        self._handler = handler

    def write_latch(
        self,
        event: StateEvent,
        class_name: str,
        score: float,
        mel_stats: Optional[dict] = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event.type.value,
            "class": class_name,
            "score": float(score),
            "latch_duration_seconds": float(event.latch_duration_seconds),
        }
        if mel_stats:
            payload["mel_stats"] = mel_stats
        self._logger.info("detection", extra={"payload": payload})

    @property
    def path(self) -> Path:
        return self._path

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def close(self) -> None:
        self._handler.flush()
        self._handler.close()
        self._logger.removeHandler(self._handler)
