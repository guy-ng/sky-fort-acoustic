"""Parabolic sub-grid interpolation for 2D SRP beamforming maps.

Refines the peak DOA estimate from grid-quantized to sub-degree accuracy
by fitting a parabola to the three samples around the peak along each axis.
This is BF-12: sub-degree DOA accuracy via parabolic interpolation.
"""

from __future__ import annotations

import numpy as np


def parabolic_interpolation_2d(
    srp_map: np.ndarray,
    az_idx: int,
    el_idx: int,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
) -> tuple[float, float]:
    """Refine a peak position in a 2D SRP map using parabolic interpolation.

    For each axis (azimuth, elevation), fits a parabola through the peak
    and its two neighbors to find the sub-grid maximum. At grid boundaries,
    falls back to the grid-quantized value.

    Args:
        srp_map: 2D spatial power map, shape (n_az, n_el).
        az_idx: Azimuth index of the peak in srp_map.
        el_idx: Elevation index of the peak in srp_map.
        az_grid_deg: 1D array of azimuth angles in degrees.
        el_grid_deg: 1D array of elevation angles in degrees.

    Returns:
        Tuple of (refined_az_deg, refined_el_deg).
    """
    n_az, n_el = srp_map.shape

    # Azimuth refinement
    if 0 < az_idx < n_az - 1:
        y_l = srp_map[az_idx - 1, el_idx]
        y_c = srp_map[az_idx, el_idx]
        y_r = srp_map[az_idx + 1, el_idx]
        denom = y_l - 2.0 * y_c + y_r
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_l - y_r) / denom
            az_step = float(az_grid_deg[1] - az_grid_deg[0])
            az_refined = float(az_grid_deg[az_idx]) + delta * az_step
        else:
            az_refined = float(az_grid_deg[az_idx])
    else:
        az_refined = float(az_grid_deg[az_idx])

    # Elevation refinement
    if 0 < el_idx < n_el - 1:
        y_l = srp_map[az_idx, el_idx - 1]
        y_c = srp_map[az_idx, el_idx]
        y_r = srp_map[az_idx, el_idx + 1]
        denom = y_l - 2.0 * y_c + y_r
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_l - y_r) / denom
            el_step = float(el_grid_deg[1] - el_grid_deg[0])
            el_refined = float(el_grid_deg[el_idx]) + delta * el_step
        else:
            el_refined = float(el_grid_deg[el_idx])
    else:
        el_refined = float(el_grid_deg[el_idx])

    return az_refined, el_refined
