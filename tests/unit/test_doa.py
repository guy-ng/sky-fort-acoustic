"""Unit tests for DOA coordinate transform module (DOA-01, DOA-02)."""

import pytest

from acoustic.tracking.doa import MountingOrientation, array_to_world


class TestArrayToWorld:
    """Tests for array_to_world coordinate transform."""

    def test_broadside_identity_vertical(self):
        """DOA-02 / D-10: broadside (0,0) must produce pan=0, tilt=0."""
        pan, tilt = array_to_world(0.0, 0.0, MountingOrientation.VERTICAL_Y_UP)
        assert pan == 0.0
        assert tilt == 0.0

    def test_positive_pan_right_tilt_up(self):
        """D-09: positive az -> positive pan (right), positive el -> positive tilt (up)."""
        pan, tilt = array_to_world(30.0, 10.0, MountingOrientation.VERTICAL_Y_UP)
        assert pan == 30.0
        assert tilt == 10.0

    def test_negative_directions(self):
        """Negative az/el preserved as negative pan/tilt."""
        pan, tilt = array_to_world(-45.0, -20.0, MountingOrientation.VERTICAL_Y_UP)
        assert pan == -45.0
        assert tilt == -20.0

    def test_horizontal_mount_broadside(self):
        """Horizontal mount broadside identity."""
        pan, tilt = array_to_world(0.0, 0.0, MountingOrientation.HORIZONTAL)
        assert pan == 0.0
        assert tilt == 0.0

    def test_unknown_mounting_raises(self):
        """Unknown mounting orientation must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mounting"):
            array_to_world(10.0, 5.0, "invalid")

    def test_enum_values(self):
        """MountingOrientation enum has expected string values."""
        assert MountingOrientation.VERTICAL_Y_UP.value == "vertical_y_up"
        assert MountingOrientation.HORIZONTAL.value == "horizontal"
