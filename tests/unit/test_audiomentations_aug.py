"""Unit tests for AudiomentationsAugmentation and ComposedAugmentation."""

from __future__ import annotations

import pickle

import numpy as np
import pytest


def _sine_audio(freq: float = 440.0, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    """Generate a sine wave test signal."""
    return np.sin(2 * np.pi * freq * np.arange(int(sr * duration)) / sr).astype(np.float32)


class TestAudiomentationsAugmentation:
    """Tests for AudiomentationsAugmentation class."""

    def test_output_shape(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation

        aug = AudiomentationsAugmentation(p=1.0)
        audio = _sine_audio()
        out = aug(audio)
        assert out.ndim == 1, "Output should be 1-D"

    def test_output_dtype(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation

        aug = AudiomentationsAugmentation(p=1.0)
        audio = _sine_audio()
        out = aug(audio)
        assert out.dtype == np.float32

    def test_no_augmentation(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation

        aug = AudiomentationsAugmentation(p=0.0)
        audio = _sine_audio()
        out = aug(audio)
        assert len(out) == len(audio), "Output length should match input when p=0"

    def test_augmentation_changes_audio(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation

        aug = AudiomentationsAugmentation(p=1.0)
        audio = _sine_audio()
        changed = False
        for _ in range(10):
            out = aug(audio)
            if not np.array_equal(out, audio):
                changed = True
                break
        assert changed, "With p=1.0, augmentation should change audio in at least one trial"

    def test_pitch_shift_preserves_length(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation

        aug = AudiomentationsAugmentation(p=1.0)
        audio = _sine_audio()
        out = aug(audio)
        assert len(out) == len(audio), "Pitch shift should preserve audio length"


class TestComposedAugmentation:
    """Tests for ComposedAugmentation class."""

    def test_composed_augmentation_picklable(self):
        from acoustic.training.augmentation import AudiomentationsAugmentation, ComposedAugmentation

        aug1 = AudiomentationsAugmentation(p=0.0)
        aug2 = AudiomentationsAugmentation(p=0.0)
        composed = ComposedAugmentation(augmentations=[aug1, aug2])

        # Pickle round-trip
        data = pickle.dumps(composed)
        restored = pickle.loads(data)

        audio = _sine_audio()
        out = restored(audio)
        assert out.dtype == np.float32
        assert len(out) == len(audio)

    def test_composed_augmentation_chains_correctly(self):
        from acoustic.training.augmentation import ComposedAugmentation

        # Two simple augmentations: multiply by 2, then add 1
        class MulTwo:
            def __call__(self, audio: np.ndarray) -> np.ndarray:
                return audio * 2.0

        class AddOne:
            def __call__(self, audio: np.ndarray) -> np.ndarray:
                return audio + 1.0

        composed = ComposedAugmentation(augmentations=[MulTwo(), AddOne()])
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = composed(audio)
        expected = audio * 2.0 + 1.0
        np.testing.assert_array_almost_equal(out, expected)
