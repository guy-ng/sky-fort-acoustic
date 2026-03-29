"""UMA-16v2 microphone geometry and 2D steering vector generation.

Ported from POC: radar_gui_all_mics_fast_drone.py build_mic_positions (lines 26-72).
Extended with 2D steering vectors per RESEARCH Pattern 2.
"""

import numpy as np

# UMA-16v2 array constants
SPACING = 0.042  # meters (42mm between adjacent mics)
NUM_CHANNELS = 16


def build_mic_positions() -> np.ndarray:
    """Build microphone positions for UMA-16v2.

    USB channels: ch0 -> MIC1, ch1 -> MIC2, ..., ch15 -> MIC16

    Mechanical drawing (top view):
        Row 0 (top):    MIC8   MIC7   MIC10  MIC9
        Row 1:          MIC6   MIC5   MIC12  MIC11
        Row 2:          MIC4   MIC3   MIC14  MIC13
        Row 3 (bottom): MIC2   MIC1   MIC16  MIC15

    Coordinate system:
        x: left (-) to right (+)
        y: back (-) to front (+)  (top row is +y)
        z: all zeros (planar array)

    Returns:
        np.ndarray: shape (3, 16) -- [x, y, z] for each channel
    """
    # MIC number -> (row, col) in the 4x4 grid
    mic_rc = {
        8: (0, 0), 7: (0, 1), 10: (0, 2), 9: (0, 3),
        6: (1, 0), 5: (1, 1), 12: (1, 2), 11: (1, 3),
        4: (2, 0), 3: (2, 1), 14: (2, 2), 13: (2, 3),
        2: (3, 0), 1: (3, 1), 16: (3, 2), 15: (3, 3),
    }

    xs = np.array([-1.5, -0.5, 0.5, 1.5]) * SPACING
    ys = np.array([+1.5, +0.5, -0.5, -1.5]) * SPACING  # top row is +y

    xs_all = []
    ys_all = []
    zs_all = []

    for ch in range(NUM_CHANNELS):
        mic_num = ch + 1
        row, col = mic_rc[mic_num]
        xs_all.append(xs[col])
        ys_all.append(ys[row])
        zs_all.append(0.0)

    return np.vstack([
        np.array(xs_all),
        np.array(ys_all),
        np.array(zs_all),
    ])


def build_steering_vectors_2d(
    az_grid_deg: np.ndarray, el_grid_deg: np.ndarray
) -> np.ndarray:
    """Build unit direction vectors for 2D (azimuth, elevation) grid.

    Spherical to Cartesian convention:
        - Azimuth measured from y-axis (broadside) in the xy-plane
        - Elevation measured from xy-plane upward
        - At az=0, el=0: direction = (0, 1, 0) (pointing along y-axis)

    Args:
        az_grid_deg: 1D array of azimuth angles in degrees
        el_grid_deg: 1D array of elevation angles in degrees

    Returns:
        np.ndarray: shape (n_az * n_el, 3) array of unit direction vectors
    """
    az_rad = np.deg2rad(az_grid_deg)
    el_rad = np.deg2rad(el_grid_deg)

    # Meshgrid: all combinations (indexing='ij' -> az varies first)
    az_mesh, el_mesh = np.meshgrid(az_rad, el_rad, indexing="ij")
    az_flat = az_mesh.ravel()
    el_flat = el_mesh.ravel()

    # Spherical to Cartesian
    dirs = np.stack(
        [
            np.sin(az_flat) * np.cos(el_flat),  # x
            np.cos(az_flat) * np.cos(el_flat),  # y
            np.sin(el_flat),                     # z
        ],
        axis=1,
    )

    return dirs
