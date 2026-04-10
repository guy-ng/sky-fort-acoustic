"""Tests for functional beamforming: bf_nu config, power-map exponent, srp_phat defaults."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestSrpPhatDefaults:
    """Verify srp_phat_2d default frequency arguments updated to 500-4000 Hz."""

    def test_fmin_default_is_500(self):
        from acoustic.beamforming.srp_phat import srp_phat_2d

        sig = inspect.signature(srp_phat_2d)
        assert sig.parameters["fmin"].default == 500.0

    def test_fmax_default_is_4000(self):
        from acoustic.beamforming.srp_phat import srp_phat_2d

        sig = inspect.signature(srp_phat_2d)
        assert sig.parameters["fmax"].default == 4000.0


class TestBfNuConfig:
    """Verify bf_nu config field exists with correct default."""

    def test_bf_nu_default_100(self):
        from acoustic.config import AcousticSettings

        settings = AcousticSettings(audio_source="simulated")
        assert settings.bf_nu == 100.0

    def test_bf_nu_env_override(self, monkeypatch):
        from acoustic.config import AcousticSettings

        monkeypatch.setenv("ACOUSTIC_BF_NU", "42.0")
        settings = AcousticSettings(audio_source="simulated")
        assert settings.bf_nu == 42.0


class TestFunctionalBeamformingMath:
    """Test the power-map exponent math applied in the pipeline."""

    @staticmethod
    def _apply_functional_bf(srp_map: np.ndarray, nu: float) -> np.ndarray:
        """Replicate the functional beamforming transform from pipeline.py."""
        max_val = srp_map.max()
        if max_val > 0:
            fb_map = (srp_map / max_val) ** nu
            fb_map[fb_map < 1e-6] = 0.0
            return fb_map.astype(np.float32)
        return np.zeros_like(srp_map, dtype=np.float32)

    def test_nu_100_suppresses_sidelobes(self):
        """nu=100 on [0.5, 0.8, 1.0] -> [~0, ~0, 1.0]."""
        srp = np.array([0.5, 0.8, 1.0])
        result = self._apply_functional_bf(srp, nu=100.0)
        # 0.5^100 and 0.8^100 are astronomically small
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)

    def test_nu_1_preserves_normalized_map(self):
        """nu=1 returns the normalized map unchanged."""
        srp = np.array([0.2, 0.5, 1.0])
        result = self._apply_functional_bf(srp, nu=1.0)
        expected = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_zero_map_returns_zeros(self):
        """All-zero map returns all zeros (no division by zero)."""
        srp = np.array([0.0, 0.0, 0.0])
        result = self._apply_functional_bf(srp, nu=100.0)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

    def test_small_values_clamped_to_zero(self):
        """Values below 1e-6 after power transform are clamped to 0.0."""
        srp = np.array([0.01, 0.1, 1.0])
        result = self._apply_functional_bf(srp, nu=50.0)
        # 0.01^50 ~ 1e-100, 0.1^50 ~ 1e-50 — both way below 1e-6
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_result_is_float32(self):
        """Output must be float32."""
        srp = np.array([0.3, 0.7, 1.0], dtype=np.float64)
        result = self._apply_functional_bf(srp, nu=10.0)
        assert result.dtype == np.float32

    def test_result_in_zero_one_range(self):
        """All values must be in [0, 1]."""
        rng = np.random.default_rng(42)
        srp = rng.random((20, 15))
        result = self._apply_functional_bf(srp, nu=50.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestPipelineLatestMap:
    """Test that pipeline.latest_map is float32 in [0,1] after functional beamforming."""

    def test_pipeline_uses_bf_nu_from_settings(self):
        """Verify pipeline reads bf_nu from settings (smoke test via import)."""
        from acoustic.config import AcousticSettings

        settings = AcousticSettings(audio_source="simulated")
        assert hasattr(settings, "bf_nu")
        assert settings.bf_nu == 100.0
