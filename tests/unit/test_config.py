"""Tests for AcousticSettings configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from acoustic.config import AcousticSettings


class TestAcousticSettingsDefaults:
    """Verify AcousticSettings returns correct defaults with no env vars."""

    def setup_method(self):
        """Clear any ACOUSTIC_ env vars that may leak from other test modules."""
        self._saved = {
            k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ACOUSTIC_")
        }

    def teardown_method(self):
        os.environ.update(self._saved)

    def test_default_sample_rate(self):
        s = AcousticSettings()
        assert s.sample_rate == 48000

    def test_default_num_channels(self):
        s = AcousticSettings()
        assert s.num_channels == 16

    def test_default_chunk_seconds(self):
        s = AcousticSettings()
        assert s.chunk_seconds == 0.15

    def test_default_freq_min(self):
        s = AcousticSettings()
        assert s.freq_min == 100.0

    def test_default_freq_max(self):
        s = AcousticSettings()
        assert s.freq_max == 2000.0

    def test_default_az_range(self):
        s = AcousticSettings()
        assert s.az_range == 90.0

    def test_default_el_range(self):
        s = AcousticSettings()
        assert s.el_range == 45.0

    def test_default_az_resolution(self):
        s = AcousticSettings()
        assert s.az_resolution == 1.0

    def test_default_el_resolution(self):
        s = AcousticSettings()
        assert s.el_resolution == 1.0

    def test_default_noise_percentile(self):
        s = AcousticSettings()
        assert s.noise_percentile == 95.0

    def test_default_noise_margin(self):
        s = AcousticSettings()
        assert s.noise_margin == 1.5

    def test_default_host(self):
        s = AcousticSettings()
        assert s.host == "0.0.0.0"

    def test_default_port(self):
        s = AcousticSettings()
        assert s.port == 8000

    def test_default_audio_source(self):
        s = AcousticSettings()
        assert s.audio_source == "hardware"

    def test_default_audio_device(self):
        s = AcousticSettings()
        assert s.audio_device is None

    def test_default_speed_of_sound(self):
        s = AcousticSettings()
        assert s.speed_of_sound == 343.0


class TestAcousticSettingsEnvOverride:
    """Verify environment variable overrides work with ACOUSTIC_ prefix."""

    def test_env_override_sample_rate(self):
        with patch.dict(os.environ, {"ACOUSTIC_SAMPLE_RATE": "44100"}):
            s = AcousticSettings()
            assert s.sample_rate == 44100

    def test_env_override_audio_source(self):
        with patch.dict(os.environ, {"ACOUSTIC_AUDIO_SOURCE": "simulated"}):
            s = AcousticSettings()
            assert s.audio_source == "simulated"


class TestAcousticSettingsProperties:
    """Verify computed properties."""

    def test_chunk_samples_default(self):
        s = AcousticSettings()
        assert s.chunk_samples == 7200  # int(48000 * 0.15)

    def test_ring_chunks_default(self):
        s = AcousticSettings()
        assert s.ring_chunks == 14  # ceil(2.0 / 0.15) = ceil(13.33) = 14

    def test_chunk_samples_custom(self):
        with patch.dict(os.environ, {"ACOUSTIC_SAMPLE_RATE": "44100"}):
            s = AcousticSettings()
            assert s.chunk_samples == int(44100 * 0.15)

    def test_ring_chunks_custom_chunk_seconds(self):
        with patch.dict(os.environ, {"ACOUSTIC_CHUNK_SECONDS": "0.2"}):
            s = AcousticSettings()
            assert s.ring_chunks == 10  # ceil(2.0 / 0.2) = 10


class TestAggregationConfigDefaults:
    """Verify aggregation weight defaults."""

    def setup_method(self):
        self._saved = {
            k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ACOUSTIC_")
        }

    def teardown_method(self):
        os.environ.update(self._saved)

    def test_default_agg_w_max(self):
        s = AcousticSettings()
        assert s.cnn_agg_w_max == 0.5

    def test_default_agg_w_mean(self):
        s = AcousticSettings()
        assert s.cnn_agg_w_mean == 0.5


class TestAggregationConfigEnvOverride:
    """Verify env var override for aggregation weights."""

    def test_env_override_agg_w_max(self):
        with patch.dict(os.environ, {"ACOUSTIC_CNN_AGG_W_MAX": "0.7"}):
            s = AcousticSettings()
            assert s.cnn_agg_w_max == 0.7

    def test_env_override_agg_w_mean(self):
        with patch.dict(os.environ, {"ACOUSTIC_CNN_AGG_W_MEAN": "0.3"}):
            s = AcousticSettings()
            assert s.cnn_agg_w_mean == 0.3


class TestModelTypeConfig:
    """Verify cnn_model_type config field for model architecture selection."""

    def setup_method(self):
        self._saved = {
            k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ACOUSTIC_")
        }

    def teardown_method(self):
        os.environ.update(self._saved)

    def test_model_type_config_default(self):
        """cnn_model_type defaults to 'research_cnn'."""
        s = AcousticSettings()
        assert s.cnn_model_type == "research_cnn"

    def test_model_type_config_env_override(self):
        """ACOUSTIC_CNN_MODEL_TYPE=efficientat_mn10 overrides default."""
        with patch.dict(os.environ, {"ACOUSTIC_CNN_MODEL_TYPE": "efficientat_mn10"}):
            s = AcousticSettings()
            assert s.cnn_model_type == "efficientat_mn10"
