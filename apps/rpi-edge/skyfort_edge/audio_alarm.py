"""D-17/D-18/D-19: Optional audio alarm, disabled by default, silent-degrade on failure.

Plays a bundled alert.wav once per latch cycle on rising edge. Any failure
(missing device, PortAudio exception, unreadable file) is logged at WARNING
and swallowed — the detection pipeline must never crash because of alarm
output problems (T-21-04 scope extension).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class AudioAlarm:
    def __init__(
        self,
        enabled: bool,
        alert_wav_path: Path,
        device: Optional[str] = None,
    ) -> None:
        self._enabled = enabled
        self._path = Path(alert_wav_path)
        self._device = device
        self._is_playing = False
        self._lock = threading.Lock()
        self._audio_data: Optional[np.ndarray] = None
        self._sr: Optional[int] = None

        if not self._enabled:
            log.info("AudioAlarm disabled (D-18 default)")
            return

        try:
            import soundfile as sf

            data, sr = sf.read(str(self._path), dtype="float32")
            self._audio_data = data
            self._sr = sr
            log.info(
                "AudioAlarm loaded %s (sr=%d, frames=%d)",
                self._path,
                sr,
                len(data),
            )
        except Exception as e:
            log.warning(
                "AudioAlarm could not load %s: %s — disabling",
                self._path,
                e,
            )
            self._enabled = False

    def play(self) -> None:
        """Play once per latch cycle. No-op if disabled, already playing, or device missing."""
        if not self._enabled:
            return
        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True
        try:
            import sounddevice as sd

            sd.play(
                self._audio_data,
                self._sr,
                device=self._device,
                blocking=False,
            )
        except Exception as e:
            log.warning("AudioAlarm play failed (degraded silently): %s", e)

    def reset(self) -> None:
        """Call on falling edge to allow the next rising edge to replay."""
        with self._lock:
            self._is_playing = False
