"""Peak detection with adaptive noise threshold for beamforming maps.

Implements percentile-based calibration (D-08, BF-04): the beamforming peak
must exceed the Nth percentile of the map by a configurable margin to count
as a detection.
"""

import numpy as np

from src.acoustic.types import PeakDetection


def detect_peak_with_threshold(
    srp_map: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    percentile: float = 95.0,
    margin: float = 1.5,
) -> PeakDetection | None:
    """Detect the strongest peak in the SRP map if it exceeds the noise threshold.

    The noise threshold is computed as:
        threshold = np.percentile(srp_map, percentile) * margin

    If the maximum value in the map is below this threshold, no detection
    is returned (the signal is considered noise).

    Args:
        srp_map: 2D spatial power map, shape (n_az, n_el)
        az_grid_deg: 1D array of azimuth angles in degrees
        el_grid_deg: 1D array of elevation angles in degrees
        percentile: percentile of the map to use as noise floor (0-100)
        margin: multiplier applied to the percentile value

    Returns:
        PeakDetection with azimuth, elevation, power, and threshold, or None
        if the peak is below the noise threshold.
    """
    threshold = np.percentile(srp_map, percentile) * margin
    max_val = float(np.max(srp_map))

    if max_val < threshold:
        return None

    az_idx, el_idx = np.unravel_index(np.argmax(srp_map), srp_map.shape)

    return PeakDetection(
        az_deg=float(az_grid_deg[az_idx]),
        el_deg=float(el_grid_deg[el_idx]),
        power=max_val,
        threshold=float(threshold),
    )
