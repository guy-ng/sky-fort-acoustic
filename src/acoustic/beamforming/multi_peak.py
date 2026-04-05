"""Multi-peak detection for SRP beamforming maps (BF-13).

Finds multiple simultaneous drone sources in the SRP spatial power map
by applying a noise-floor-relative threshold and greedy angular separation
constraint. Returns peaks sorted by power descending.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic.types import PeakDetection


def detect_multi_peak(
    srp_map: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    noise_floor: np.ndarray,
    threshold_factor: float = 3.0,
    min_separation_deg: float = 15.0,
    max_peaks: int = 5,
) -> list[PeakDetection]:
    """Detect multiple peaks in an SRP map above an adaptive noise threshold.

    Uses a greedy algorithm: select the strongest candidate first, then
    accept additional candidates only if they are at least
    ``min_separation_deg`` away from all previously accepted peaks.

    Args:
        srp_map: 2D spatial power map, shape (n_az, n_el).
        az_grid_deg: 1D array of azimuth angles in degrees.
        el_grid_deg: 1D array of elevation angles in degrees.
        noise_floor: 2D noise estimate with same shape as srp_map
                     (e.g. from MCRANoiseEstimator).
        threshold_factor: Multiplier on noise_floor to form detection threshold.
        min_separation_deg: Minimum angular distance between accepted peaks.
        max_peaks: Maximum number of peaks to return.

    Returns:
        List of PeakDetection sorted by power descending (strongest first).
        Empty list if no candidates exceed the threshold.
    """
    threshold = noise_floor * threshold_factor

    # Find all candidate cells above threshold
    candidates = np.argwhere(srp_map > threshold)
    if len(candidates) == 0:
        return []

    # Get power at each candidate and sort descending
    powers = np.array([srp_map[c[0], c[1]] for c in candidates])
    order = np.argsort(powers)[::-1]
    candidates = candidates[order]
    powers = powers[order]

    # Greedy selection with angular separation constraint
    peaks: list[PeakDetection] = []
    for i, c in enumerate(candidates):
        if len(peaks) >= max_peaks:
            break

        az = float(az_grid_deg[c[0]])
        el = float(el_grid_deg[c[1]])

        # Check angular distance to all already-accepted peaks
        too_close = False
        if min_separation_deg > 0:
            for p in peaks:
                dist = math.sqrt((az - p.az_deg) ** 2 + (el - p.el_deg) ** 2)
                if dist < min_separation_deg:
                    too_close = True
                    break

        if too_close:
            continue

        peaks.append(
            PeakDetection(
                az_deg=az,
                el_deg=el,
                power=float(powers[i]),
                threshold=float(threshold[c[0], c[1]]),
            )
        )

    return peaks
