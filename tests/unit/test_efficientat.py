"""Unit tests for EfficientAT mn10 model integration.

Covers: model loading, parameter count, output shape, Classifier protocol,
registry integration, and config support.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import torch

from acoustic.classification.protocols import Classifier


class TestEfficientATModel:
    """Test the vendored EfficientAT MN model architecture."""

    def test_model_loads_with_527_classes(self):
        """get_model(num_classes=527) returns a model with ~4.88M params."""
        from acoustic.classification.efficientat.model import get_model

        model = get_model(num_classes=527, width_mult=1.0)
        assert model is not None

    def test_param_count(self):
        """mn10 (width_mult=1.0) has ~4.88M parameters."""
        from acoustic.classification.efficientat.model import get_model

        model = get_model(num_classes=527, width_mult=1.0)
        count = sum(p.numel() for p in model.parameters())
        assert 4_500_000 < count < 5_500_000, f"Expected ~4.88M params, got {count:,}"

    def test_output_shape_binary(self):
        """get_model(num_classes=1) produces output shape (1, 1) for (1, 32000) input."""
        from acoustic.classification.efficientat.model import get_model
        from acoustic.classification.efficientat.preprocess import AugmentMelSTFT

        model = get_model(
            num_classes=1, width_mult=1.0, head_type="mlp",
            input_dim_f=128, input_dim_t=100,
        )
        model.eval()
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024)
        mel.eval()

        waveform = torch.randn(1, 32000)
        with torch.no_grad():
            spec = mel(waveform)
            # AugmentMelSTFT outputs (batch, n_mels, time); model needs (batch, 1, n_mels, time)
            spec = spec.unsqueeze(1)
            out, features = model(spec)
        # With batch=1 and num_classes=1, squeeze may remove all dims.
        # The important thing is that the output is a single scalar value.
        assert out.numel() == 1, f"Expected 1 element, got {out.numel()} with shape {out.shape}"


class TestEfficientATClassifier:
    """Test the EfficientATClassifier wrapper satisfies Classifier protocol."""

    def test_classifier_protocol(self):
        """EfficientATClassifier is an instance of the Classifier protocol."""
        from acoustic.classification.efficientat.classifier import EfficientATClassifier
        from acoustic.classification.efficientat.model import get_model

        model = get_model(num_classes=1, width_mult=1.0, head_type="mlp",
                          input_dim_f=128, input_dim_t=100)
        clf = EfficientATClassifier(model)
        assert isinstance(clf, Classifier)

    def test_predict_returns_float(self):
        """predict() returns a float between 0.0 and 1.0."""
        from acoustic.classification.efficientat.classifier import EfficientATClassifier
        from acoustic.classification.efficientat.model import get_model

        model = get_model(num_classes=1, width_mult=1.0, head_type="mlp",
                          input_dim_f=128, input_dim_t=100)
        clf = EfficientATClassifier(model)

        # Pass raw waveform at 32kHz (1 second)
        waveform = torch.randn(1, 32000)
        prob = clf.predict(waveform)
        assert isinstance(prob, float), f"Expected float, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Expected [0, 1], got {prob}"


class TestEfficientATRegistry:
    """Test model registry integration."""

    def test_registry_contains_efficientat(self):
        """Importing the efficientat package registers 'efficientat_mn10'."""
        import acoustic.classification.efficientat  # noqa: F401
        from acoustic.classification.ensemble import get_registered_types

        assert "efficientat_mn10" in get_registered_types()

    def test_registry_load_fails_without_checkpoint(self):
        """load_model('efficientat_mn10', ...) raises on missing file."""
        import acoustic.classification.efficientat  # noqa: F401
        from acoustic.classification.ensemble import load_model

        import pytest
        with pytest.raises(Exception):
            load_model("efficientat_mn10", "/nonexistent/checkpoint.pt")


class TestEfficientATConfig:
    """Test EfficientATMelConfig defaults."""

    def test_mel_config_defaults(self):
        from acoustic.classification.efficientat.config import EfficientATMelConfig

        cfg = EfficientATMelConfig()
        assert cfg.sample_rate == 32000
        assert cfg.n_mels == 128
        assert cfg.win_length == 800
        assert cfg.hop_size == 320
        assert cfg.n_fft == 1024
        assert cfg.input_dim_t == 100

    def test_segment_samples(self):
        from acoustic.classification.efficientat.config import EfficientATMelConfig

        cfg = EfficientATMelConfig()
        assert cfg.segment_samples == 32000  # 100 * 320


class TestAcousticSettingsModelType:
    """Test cnn_model_type config field."""

    def setup_method(self):
        self._saved = {
            k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ACOUSTIC_")
        }

    def teardown_method(self):
        os.environ.update(self._saved)

    def test_default_model_type(self):
        from acoustic.config import AcousticSettings
        s = AcousticSettings()
        assert s.cnn_model_type == "research_cnn"

    def test_env_override_model_type(self):
        with patch.dict(os.environ, {"ACOUSTIC_CNN_MODEL_TYPE": "efficientat_mn10"}):
            from acoustic.config import AcousticSettings
            s = AcousticSettings()
            assert s.cnn_model_type == "efficientat_mn10"
