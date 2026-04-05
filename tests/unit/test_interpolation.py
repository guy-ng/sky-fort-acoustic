"""Unit tests for parabolic sub-grid interpolation of SRP maps."""

import numpy as np
import pytest

from acoustic.beamforming.interpolation import parabolic_interpolation_2d


class TestParabolicInterpolation2D:
    """Tests for the parabolic_interpolation_2d function."""

    def test_center_peak_refines_to_subgrid(self):
        """Test 1: Peak at center of grid with Gaussian-like neighbors refines to sub-grid position."""
        az_grid = np.arange(-10, 11, 1.0)  # 21 points, 1-deg spacing
        el_grid = np.arange(-5, 6, 1.0)    # 11 points, 1-deg spacing
        srp_map = np.zeros((len(az_grid), len(el_grid)))
        # Place a Gaussian-like peak shifted slightly from grid center
        az_idx, el_idx = 10, 5  # center
        srp_map[az_idx, el_idx] = 10.0
        srp_map[az_idx - 1, el_idx] = 7.0
        srp_map[az_idx + 1, el_idx] = 8.0  # asymmetric -> sub-grid shift
        srp_map[az_idx, el_idx - 1] = 6.0
        srp_map[az_idx, el_idx + 1] = 9.0  # asymmetric -> sub-grid shift

        az_ref, el_ref = parabolic_interpolation_2d(srp_map, az_idx, el_idx, az_grid, el_grid)
        # Should differ from grid-quantized by a fractional amount
        assert az_ref != az_grid[az_idx], "Azimuth should be refined from grid value"
        assert el_ref != el_grid[el_idx], "Elevation should be refined from grid value"
        # Should be close to grid value (within one step)
        assert abs(az_ref - az_grid[az_idx]) < 1.0
        assert abs(el_ref - el_grid[el_idx]) < 1.0

    def test_boundary_peak_returns_grid_value(self):
        """Test 2: Peak at grid boundary returns grid-quantized value."""
        az_grid = np.arange(0, 10, 1.0)
        el_grid = np.arange(0, 10, 1.0)
        srp_map = np.random.rand(len(az_grid), len(el_grid))

        # Test az boundary (az_idx=0)
        az_ref, el_ref = parabolic_interpolation_2d(srp_map, 0, 5, az_grid, el_grid)
        assert az_ref == az_grid[0], "Boundary az should return grid value"

        # Test az boundary (az_idx=n_az-1)
        az_ref, el_ref = parabolic_interpolation_2d(
            srp_map, len(az_grid) - 1, 5, az_grid, el_grid
        )
        assert az_ref == az_grid[-1], "Boundary az should return grid value"

        # Test el boundary (el_idx=0)
        az_ref, el_ref = parabolic_interpolation_2d(srp_map, 5, 0, az_grid, el_grid)
        assert el_ref == el_grid[0], "Boundary el should return grid value"

        # Test el boundary (el_idx=n_el-1)
        az_ref, el_ref = parabolic_interpolation_2d(
            srp_map, 5, len(el_grid) - 1, az_grid, el_grid
        )
        assert el_ref == el_grid[-1], "Boundary el should return grid value"

    def test_symmetric_neighbors_zero_delta(self):
        """Test 3: Symmetric neighbors produce zero delta."""
        az_grid = np.arange(-5, 6, 1.0)
        el_grid = np.arange(-5, 6, 1.0)
        srp_map = np.zeros((len(az_grid), len(el_grid)))
        az_idx, el_idx = 5, 5
        srp_map[az_idx, el_idx] = 10.0
        srp_map[az_idx - 1, el_idx] = 5.0
        srp_map[az_idx + 1, el_idx] = 5.0  # symmetric
        srp_map[az_idx, el_idx - 1] = 5.0
        srp_map[az_idx, el_idx + 1] = 5.0  # symmetric

        az_ref, el_ref = parabolic_interpolation_2d(srp_map, az_idx, el_idx, az_grid, el_grid)
        assert az_ref == pytest.approx(az_grid[az_idx], abs=1e-10)
        assert el_ref == pytest.approx(el_grid[el_idx], abs=1e-10)

    def test_known_analytical_case(self):
        """Test 4: Known analytical case -- delta = 1/6 degree for values (1.0, 3.0, 2.0)."""
        az_grid = np.arange(0, 10, 1.0)  # 1-degree spacing
        el_grid = np.arange(0, 10, 1.0)
        srp_map = np.zeros((len(az_grid), len(el_grid)))
        k = 5
        el_k = 5
        # Set azimuth neighbors: (1.0, 3.0, 2.0)
        srp_map[k - 1, el_k] = 1.0
        srp_map[k, el_k] = 3.0
        srp_map[k + 1, el_k] = 2.0
        # Set symmetric el neighbors so el doesn't shift
        srp_map[k, el_k - 1] = 1.0
        srp_map[k, el_k + 1] = 1.0

        az_ref, el_ref = parabolic_interpolation_2d(srp_map, k, el_k, az_grid, el_grid)
        # delta = 0.5 * (y_l - y_r) / (y_l - 2*y_c + y_r)
        # = 0.5 * (1.0 - 2.0) / (1.0 - 6.0 + 2.0) = 0.5 * (-1) / (-3) = 1/6
        expected_az = az_grid[k] + (1.0 / 6.0) * 1.0  # step = 1.0 degree
        assert az_ref == pytest.approx(expected_az, abs=1e-10)

    def test_independent_az_el_refinement(self):
        """Test 5: Works correctly on both azimuth and elevation axes independently."""
        az_grid = np.arange(0, 10, 2.0)  # 2-degree spacing
        el_grid = np.arange(0, 10, 3.0)  # 3-degree spacing (different from az)
        srp_map = np.zeros((len(az_grid), len(el_grid)))
        az_idx, el_idx = 2, 1
        # Azimuth: (2.0, 5.0, 3.0) -> delta = 0.5*(2-3)/(2-10+3) = 0.5*(-1)/(-5) = 0.1
        srp_map[az_idx - 1, el_idx] = 2.0
        srp_map[az_idx, el_idx] = 5.0
        srp_map[az_idx + 1, el_idx] = 3.0
        # Elevation: (1.0, 5.0, 4.0) -> delta = 0.5*(1-4)/(1-10+4) = 0.5*(-3)/(-5) = 0.3
        srp_map[az_idx, el_idx - 1] = 1.0
        srp_map[az_idx, el_idx + 1] = 4.0

        az_ref, el_ref = parabolic_interpolation_2d(srp_map, az_idx, el_idx, az_grid, el_grid)
        expected_az = az_grid[az_idx] + 0.1 * 2.0  # delta * step
        expected_el = el_grid[el_idx] + 0.3 * 3.0  # delta * step
        assert az_ref == pytest.approx(expected_az, abs=1e-10)
        assert el_ref == pytest.approx(expected_el, abs=1e-10)
