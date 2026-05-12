"""Tests for UMA-16v2 device detection."""

from __future__ import annotations

from unittest.mock import patch

from acoustic.audio.device import detect_audio_device, detect_uma16v2
from acoustic.types import DeviceInfo


class TestDetectUma16v2:
    """Device detection tests with mocked sounddevice."""

    def test_detect_uma16v2_found(self):
        mock_devices = [
            {
                "name": "UMA16v2: USB Audio (hw:2,0)",
                "max_input_channels": 16,
                "default_samplerate": 48000.0,
            }
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            result = detect_uma16v2()
            assert result is not None
            assert isinstance(result, DeviceInfo)
            assert result.index == 0
            assert result.name == "UMA16v2: USB Audio (hw:2,0)"
            assert result.channels == 16
            assert result.default_samplerate == 48000.0

    def test_detect_uma16v2_not_found(self):
        mock_devices = [
            {
                "name": "Built-in Mic",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            }
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            result = detect_uma16v2()
            assert result is None

    def test_detect_uma16v2_case_insensitive(self):
        mock_devices = [
            {
                "name": "uma16v2: USB Audio",
                "max_input_channels": 16,
                "default_samplerate": 48000.0,
            }
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            result = detect_uma16v2()
            assert result is not None
            assert result.name == "uma16v2: USB Audio"


class TestDetectAudioDevice:
    """`detect_audio_device` returns the UMA-16v2 or nothing — no fallback."""

    def test_returns_uma16v2_when_present(self):
        mock_devices = [
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "default_samplerate": 44100.0,
            },
            {
                "name": "UMA16v2: USB Audio",
                "max_input_channels": 16,
                "default_samplerate": 48000.0,
            },
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            result = detect_audio_device()
            assert isinstance(result, DeviceInfo)
            assert result.index == 1
            assert result.name == "UMA16v2: USB Audio"
            assert result.channels == 16

    def test_returns_none_when_only_other_input_devices(self):
        mock_devices = [
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "default_samplerate": 44100.0,
            },
            {
                "name": "ReSpeaker 4 Mic Array (UAC1.0)",
                "max_input_channels": 6,
                "default_samplerate": 16000.0,
            },
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            assert detect_audio_device() is None

    def test_returns_none_when_no_input_devices(self):
        mock_devices = [
            {
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "default_samplerate": 44100.0,
            },
        ]
        with patch("acoustic.audio.device.sd.query_devices", return_value=mock_devices):
            assert detect_audio_device() is None
