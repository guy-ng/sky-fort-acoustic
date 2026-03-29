"""Tests for mic geometry and 2D steering vectors."""

import numpy as np
import numpy.testing as npt

from src.acoustic.beamforming.geometry import (
    SPACING,
    build_mic_positions,
    build_steering_vectors_2d,
)


class TestBuildMicPositions:
    def test_mic_positions_shape(self):
        """build_mic_positions() returns shape (3, 16)."""
        R = build_mic_positions()
        assert R.shape == (3, 16)

    def test_mic_positions_planar(self):
        """All z-coordinates are 0.0 (planar array)."""
        R = build_mic_positions()
        npt.assert_array_equal(R[2, :], 0.0)

    def test_mic_positions_spacing(self):
        """Max x distance = 3 * 0.042 = 0.126m, max y distance = 3 * 0.042 = 0.126m."""
        R = build_mic_positions()
        x_range = np.max(R[0]) - np.min(R[0])
        y_range = np.max(R[1]) - np.min(R[1])
        npt.assert_allclose(x_range, 3 * SPACING, atol=1e-10)
        npt.assert_allclose(y_range, 3 * SPACING, atol=1e-10)

    def test_mic_positions_channel_mapping(self):
        """Channel 0 (MIC1) is at row=3, col=1 per POC mapping.

        xs = [-1.5, -0.5, 0.5, 1.5] * SPACING
        ys = [+1.5, +0.5, -0.5, -1.5] * SPACING
        MIC1 -> row=3, col=1 -> x=xs[1]=-0.5*0.042, y=ys[3]=-1.5*0.042
        """
        R = build_mic_positions()
        expected_x = -0.5 * SPACING
        expected_y = -1.5 * SPACING
        npt.assert_allclose(R[0, 0], expected_x, atol=1e-10)
        npt.assert_allclose(R[1, 0], expected_y, atol=1e-10)


class TestBuildSteeringVectors2D:
    def test_steering_vectors_2d_shape(self):
        """build_steering_vectors_2d with 61 az points and 31 el points returns (61*31, 3)."""
        az = np.arange(-90, 91, 3)  # 61 points
        el = np.arange(-45, 46, 3)  # 31 points
        dirs = build_steering_vectors_2d(az, el)
        assert dirs.shape == (61 * 31, 3)

    def test_steering_vectors_2d_unit(self):
        """All direction vectors have norm 1.0 (unit vectors)."""
        az = np.arange(-90, 91, 10)
        el = np.arange(-45, 46, 10)
        dirs = build_steering_vectors_2d(az, el)
        norms = np.linalg.norm(dirs, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-10)

    def test_steering_vectors_broadside(self):
        """At az=0, el=0, direction vector is (0, 1, 0) -- pointing along y-axis."""
        az = np.array([0.0])
        el = np.array([0.0])
        dirs = build_steering_vectors_2d(az, el)
        npt.assert_allclose(dirs[0], [0.0, 1.0, 0.0], atol=1e-10)
