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
            # ReSpeaker XMOS firmware (Seeed 0x2886) exposes channel 0 as the
            # processed mono (AGC + AEC + on-board beamformer) and channels 1-4
            # as raw mic capsules. The processed channel is out-of-distribution
            # for a CNN trained on raw UMA-16 audio and triggers false
            # positives, so when we recognize the device we pull just the
            # first raw mic instead.
            name_lower = name.lower()
            is_respeaker = (
                "respeaker" in name_lower
                or "seeed" in name_lower
                or "xmos" in name_lower
            )
            mic_channels: tuple[int, ...] | None = None
            recommended_gain: float | None = None
            highpass_hz: float | None = None
            lowpass_hz: float | None = None
            if is_respeaker and channels >= 2:
                mic_channels = (1,)
                # ReSpeaker raw mic 1 sits ~10 dB hotter than the UMA mic at
                # the same SPL and the EfficientAT calibration of 500 clips
                # immediately. 50 keeps quiet-room peaks below ~0.7 with
                # plenty of headroom for actual drone audio.
                recommended_gain = 50.0
                # ReSpeaker noise distribution (measured on a quiet desk):
                #   raw band-RMS:    0-100 Hz=1.232  100-200=0.332  200-4k=0.351
                # ~85% of total energy lives below 200 Hz (mains hum at 50/100
                # Hz + harmonics + USB power supply). 200-4000 Hz is the
                # drone-relevant band: fundamentals start ~200 Hz, harmonics
                # extend through ~4 kHz, and everything above is HVAC/ambient.
                # A 200-4000 Hz Butterworth bandpass at the capture layer
                # drops the noise floor ~10 dB relative to raw and pulls the
                # CNN input back inside the training distribution.
                highpass_hz = 200.0
                lowpass_hz = 4000.0

            fallback = DeviceInfo(
                index=idx,
                name=name,
                channels=channels,
                default_samplerate=dev.get("default_samplerate", 48000.0),
                is_fallback=True,
                mic_channels=mic_channels,
                recommended_gain=recommended_gain,
                highpass_hz=highpass_hz,
                lowpass_hz=lowpass_hz,
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
