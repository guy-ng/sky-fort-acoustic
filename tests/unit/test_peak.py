"""Tests for peak detection with adaptive noise threshold."""

import numpy as np

from acoustic.beamforming.peak import detect_peak_with_threshold
from acoustic.types import PeakDetection


def _make_map_with_peak(
    n_az: int, n_el: int, az_idx: int, el_idx: int, peak_val: float, noise_val: float
) -> np.ndarray:
    """Create a synthetic SRP map with a known peak at (az_idx, el_idx)."""
    srp_map = np.full((n_az, n_el), noise_val)
    srp_map[az_idx, el_idx] = peak_val
    return srp_map


class TestDetectPeak:
    def test_detect_peak_strong_source(self):
        """Map with clear peak at known (az, el) returns correct PeakDetection."""
        az_grid = np.arange(-90, 91, 3)  # 61 points
        el_grid = np.arange(-45, 46, 3)  # 31 points

        # Peak at az=30, el=15 -> indices: az_idx=40, el_idx=20
        az_idx = np.searchsorted(az_grid, 30)
        el_idx = np.searchsorted(el_grid, 15)

        srp_map = _make_map_with_peak(len(az_grid), len(el_grid),
                                       az_idx, el_idx, peak_val=10.0, noise_val=1.0)

        result = detect_peak_with_threshold(srp_map, az_grid, el_grid,
                                            percentile=95.0, margin=1.5)
        assert result is not None
        assert isinstance(result, PeakDetection)
        assert abs(result.az_deg - 30.0) <= 1.0
        assert abs(result.el_deg - 15.0) <= 1.0
        assert result.power == 10.0

    def test_detect_peak_below_threshold(self):
        """Uniform noise map (no peak above threshold) returns None."""
        az_grid = np.arange(-90, 91, 3)
        el_grid = np.arange(-45, 46, 3)

        # Uniform map -- max equals the percentile, so max < percentile * margin
        srp_map = np.ones((len(az_grid), len(el_grid)))

        result = detect_peak_with_threshold(srp_map, az_grid, el_grid,
                                            percentile=95.0, margin=1.5)
        assert result is None

    def test_detect_peak_noise_threshold(self):
        """With percentile=95 and margin=1.5, peak at 2x the 95th percentile
        is detected; peak at 1.1x is not."""
        az_grid = np.arange(-90, 91, 3)
        el_grid = np.arange(-45, 46, 3)
        n_az, n_el = len(az_grid), len(el_grid)

        noise_val = 1.0
        p95 = noise_val  # For a uniform map, 95th percentile ~= noise_val

        # Peak at 2x p95 -> 2.0 > 1.0 * 1.5 = 1.5 -> detected
        srp_high = _make_map_with_peak(n_az, n_el, 30, 15, peak_val=2.0, noise_val=noise_val)
        result_high = detect_peak_with_threshold(srp_high, az_grid, el_grid,
                                                  percentile=95.0, margin=1.5)
        assert result_high is not None

        # Peak at 1.1x p95 -> 1.1 < 1.0 * 1.5 = 1.5 -> not detected
        srp_low = _make_map_with_peak(n_az, n_el, 30, 15, peak_val=1.1, noise_val=noise_val)
        result_low = detect_peak_with_threshold(srp_low, az_grid, el_grid,
                                                 percentile=95.0, margin=1.5)
        assert result_low is None

    def test_detect_peak_configurable_percentile(self):
        """Changing percentile from 95 to 50 lowers the threshold, detecting weaker peaks."""
        az_grid = np.arange(-90, 91, 3)
        el_grid = np.arange(-45, 46, 3)
        n_az, n_el = len(az_grid), len(el_grid)

        # Weak peak at 1.3x noise
        srp_map = _make_map_with_peak(n_az, n_el, 30, 15, peak_val=1.3, noise_val=1.0)

        # percentile=95, margin=1.5 -> threshold ~1.5 -> 1.3 < 1.5 -> not detected
        result_95 = detect_peak_with_threshold(srp_map, az_grid, el_grid,
                                                percentile=95.0, margin=1.5)
        assert result_95 is None

        # percentile=50, margin=1.0 -> threshold ~1.0 -> 1.3 > 1.0 -> detected
        result_50 = detect_peak_with_threshold(srp_map, az_grid, el_grid,
                                                percentile=50.0, margin=1.0)
        assert result_50 is not None

    def test_origin_suppressed(self):
        """Peak at origin (0,0) is ignored when ignore_origin_deg is set."""
        srp_map = np.ones((181, 91)) * 0.01
        az_grid = np.arange(-90, 91, 1.0)
        el_grid = np.arange(-45, 46, 1.0)
        # Put strongest peak at origin
        srp_map[90, 45] = 10.0
        # Put secondary peak away from origin
        srp_map[130, 60] = 5.0

        # Without suppression: finds origin peak
        peak = detect_peak_with_threshold(srp_map, az_grid, el_grid, percentile=50, margin=1.0)
        assert peak is not None
        assert peak.az_deg == 0.0
        assert peak.el_deg == 0.0

        # With suppression: skips origin, finds secondary peak
        peak = detect_peak_with_threshold(
            srp_map, az_grid, el_grid, percentile=50, margin=1.0, ignore_origin_deg=3.5
        )
        assert peak is not None
        assert peak.az_deg == 40.0
        assert peak.el_deg == 15.0
