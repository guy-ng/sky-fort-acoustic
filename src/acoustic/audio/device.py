"""UMA-16v2 microphone array device detection."""

from __future__ import annotations

import logging

import sounddevice as sd

from acoustic.types import DeviceInfo

logger = logging.getLogger(__name__)


def detect_uma16v2() -> DeviceInfo | None:
    """Scan audio devices and find the UMA-16v2 microphone array.

    Searches sounddevice.query_devices() for a device whose name contains
    'UMA16v2' (case-insensitive) with at least 16 input channels.

    Returns:
        DeviceInfo if found, None otherwise.
    """
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev.get("name", "")
        channels = dev.get("max_input_channels", 0)
        if "uma16v2" in name.lower() and channels >= 16:
            info = DeviceInfo(
                index=idx,
                name=name,
                channels=channels,
                default_samplerate=dev.get("default_samplerate", 48000.0),
            )
            logger.info("UMA-16v2 detected: %s (index=%d, channels=%d)", name, idx, channels)
            return info

    logger.info("UMA-16v2 not detected among %d audio devices", len(devices))
    return None


def detect_audio_device() -> DeviceInfo | None:
    """Return the UMA-16v2 if present, otherwise the first available input device.

    Falls back to the first device with ``max_input_channels >= 1`` so the
    backend can still start when the UMA-16v2 is unplugged. The returned
    ``DeviceInfo.channels`` reflects the device's actual channel count, which
    callers must use when opening the audio stream (sounddevice refuses to
    open more channels than the device exposes).

    Returns:
        DeviceInfo for the UMA-16v2, the fallback device, or None if no
        input device exists.
    """
    info = detect_uma16v2()
    if info is not None:
        return info

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        channels = dev.get("max_input_channels", 0)
        if channels >= 1:
            name = dev.get("name", "")
            fallback = DeviceInfo(
                index=idx,
                name=name,
                channels=channels,
                default_samplerate=dev.get("default_samplerate", 48000.0),
                is_fallback=True,
            )
            logger.warning(
                "UMA-16v2 not found — falling back to '%s' (index=%d, channels=%d)",
                name,
                idx,
                channels,
            )
            return fallback

    logger.warning("No input audio devices available")
    return None
