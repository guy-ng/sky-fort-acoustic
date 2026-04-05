"""Unit tests for multi-peak detection in SRP beamforming maps."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic.beamforming.multi_peak import detect_multi_peak
from acoustic.types import PeakDetection


def _make_grid(n_az: int = 37, n_el: int = 19):
    """Create azimuth/elevation grids: az -90..90, el -45..45."""
    az = np.linspace(-90, 90, n_az)
    el = np.linspace(-45, 45, n_el)
    return az, el


def _make_srp_with_peaks(az_grid, el_grid, peaks: list[tuple[float, float, float]]):
    """Create an SRP map with Gaussian peaks at specified (az, el, power) locations.

    Background is uniform 1.0.
    """
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid, indexing="ij")
    srp = np.ones_like(az_mesh)
    for az_deg, el_deg, power in peaks:
        dist_sq = (az_mesh - az_deg) ** 2 + (el_mesh - el_deg) ** 2
        srp += (power - 1.0) * np.exp(-dist_sq / 4.0)  # tight Gaussian
    return srp


class TestMultiPeakDetection:
    """Tests for detect_multi_peak function."""

    def test_two_peaks_well_separated(self):
        """Two peaks separated by >15 degrees are both returned, sorted by power descending."""
        az, el = _make_grid()
        # Peak at (-45, 0) with power 20, peak at (45, 0) with power 15
        srp = _make_srp_with_peaks(az, el, [(-45, 0, 20.0), (45, 0, 15.0)])
        noise_floor = np.ones_like(srp)

        peaks = detect_multi_peak(srp, az, el, noise_floor, threshold_factor=3.0)

        assert len(peaks) == 2
        # Sorted by power descending
        assert peaks[0].power > peaks[1].power
        assert abs(peaks[0].az_deg - (-45)) < 3.0
        assert abs(peaks[1].az_deg - 45) < 3.0

    def test_two_peaks_too_close(self):
        """Two peaks separated by <15 degrees returns only the stronger peak."""
        az, el = _make_grid()
        # Peaks 10 degrees apart
        srp = _make_srp_with_peaks(az, el, [(-5, 0, 20.0), (5, 0, 15.0)])
        noise_floor = np.ones_like(srp)

        peaks = detect_multi_peak(
            srp, az, el, noise_floor, threshold_factor=3.0, min_separation_deg=15.0
        )

        assert len(peaks) == 1
        assert abs(peaks[0].az_deg - (-5)) < 3.0  # stronger peak

    def test_all_below_threshold_returns_empty(self):
        """SRP map with all values below noise_floor * threshold returns empty list."""
        az, el = _make_grid()
        srp = np.ones((len(az), len(el))) * 2.0
        noise_floor = np.ones_like(srp)  # threshold = 1.0 * 3.0 = 3.0

        peaks = detect_multi_peak(srp, az, el, noise_floor, threshold_factor=3.0)

        assert peaks == []

    def test_max_peaks_limit(self):
        """SRP map with 7 well-separated peaks returns at most max_peaks=5."""
        az, el = _make_grid()
        # 7 peaks spread across azimuth, all well separated
        peak_locs = [
            (-80, 0, 20.0),
            (-55, 0, 18.0),
            (-30, 0, 16.0),
            (0, 0, 14.0),
            (30, 0, 12.0),
            (55, 0, 10.0),
            (80, 0, 8.0),
        ]
        srp = _make_srp_with_peaks(az, el, peak_locs)
        noise_floor = np.ones_like(srp)

        peaks = detect_multi_peak(
            srp, az, el, noise_floor, threshold_factor=3.0, max_peaks=5
        )

        assert len(peaks) == 5

    def test_peak_detection_values(self):
        """Returned PeakDetection objects have correct az_deg, el_deg, power, threshold."""
        az, el = _make_grid()
        srp = _make_srp_with_peaks(az, el, [(0, 0, 15.0)])
        noise_floor = np.ones_like(srp)

        peaks = detect_multi_peak(srp, az, el, noise_floor, threshold_factor=3.0)

        assert len(peaks) == 1
        p = peaks[0]
        assert isinstance(p, PeakDetection)
        assert abs(p.az_deg) < 3.0
        assert abs(p.el_deg) < 3.0
        assert p.power > 10.0  # Should be close to 15.0
        assert p.threshold > 0.0

    def test_zero_separation_returns_all(self):
        """min_separation_deg=0 returns all peaks above threshold."""
        az, el = _make_grid()
        # Two peaks very close together
        srp = _make_srp_with_peaks(az, el, [(-5, 0, 20.0), (5, 0, 15.0)])
        noise_floor = np.ones_like(srp)

        peaks = detect_multi_peak(
            srp, az, el, noise_floor, threshold_factor=3.0, min_separation_deg=0.0
        )

        assert len(peaks) >= 2
