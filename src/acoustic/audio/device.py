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
    """Return the UMA-16v2 if connected, otherwise None.

    Beamforming requires the 16-channel UMA-16v2 array. There is no usable
    fallback — single-mic devices cannot produce a spatial map — so when the
    UMA is absent we report no device and let the UI prompt the user to
    connect it.
    """
    return detect_uma16v2()
