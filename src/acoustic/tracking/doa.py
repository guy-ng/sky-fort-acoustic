"""Direction of Arrival coordinate transform.

Converts array-frame (azimuth, elevation) to world-frame (pan, tilt) degrees,
accounting for the UMA-16v2 mounting orientation.

The array's internal coordinate system (defined in geometry.py):
  - Azimuth: from y-axis broadside in the xy-plane, positive = rightward (+x)
  - Elevation: from xy-plane, positive = upward (+z)
  - At az=0, el=0: direction = (0, 1, 0) — broadside, directly forward

World-frame output convention (D-08, D-09):
  - pan=0, tilt=0 = broadside center (directly in front of the array)
  - pan positive = target to the right (looking from behind array)
  - tilt positive = target above horizontal

For VERTICAL_Y_UP mounting (D-01), the array's convention already matches
the world convention — the transform is identity.
"""

from __future__ import annotations

from enum import Enum


class MountingOrientation(str, Enum):
    """Physical mounting orientation of the UMA-16v2 array."""

    VERTICAL_Y_UP = "vertical_y_up"
    HORIZONTAL = "horizontal"


def array_to_world(
    az_deg: float,
    el_deg: float,
    mounting: MountingOrientation = MountingOrientation.VERTICAL_Y_UP,
) -> tuple[float, float]:
    """Convert array-frame (az, el) to world-frame (pan, tilt).

    Args:
        az_deg: Array azimuth in degrees (from broadside, positive = right).
        el_deg: Array elevation in degrees (from horizontal, positive = up).
        mounting: Physical mounting orientation of the array.

    Returns:
        Tuple of (pan_deg, tilt_deg) in world coordinates.

    Raises:
        ValueError: If mounting orientation is not a known MountingOrientation.
    """
    if mounting == MountingOrientation.VERTICAL_Y_UP:
        # Array az: from y-axis broadside, positive = rightward (x+)
        # Array el: from xy-plane, positive = upward (z+)
        # Both already match world convention (D-08, D-09).
        return az_deg, el_deg
    elif mounting == MountingOrientation.HORIZONTAL:
        # Same mapping; note that elevation accuracy is poor with a planar
        # array in horizontal mount (zero z-baseline).
        return az_deg, el_deg
    raise ValueError(f"Unknown mounting: {mounting}")
