# Phase 17: Beamforming Engine Upgrade and Pipeline Integration - Research

**Researched:** 2026-04-05
**Domain:** Real-time SRP-PHAT beamforming, DSP filtering, noise estimation, pipeline integration
**Confidence:** HIGH

## Summary

Phase 17 upgrades the existing SRP-PHAT beamforming engine from its current stubbed-out state back into the live pipeline, adding research-validated parameters: a 500-4000 Hz bandpass pre-filter, sub-grid parabolic interpolation for sub-degree DOA accuracy, multi-peak detection for simultaneous sources, and MCRA adaptive noise estimation. The beamforming is also made demand-driven -- activating only when the CNN detects a drone, deactivating after 5 seconds of silence.

The existing codebase already has a complete SRP-PHAT implementation in `src/acoustic/beamforming/` (geometry, gcc_phat, srp_phat, peak detection) that was built in Phase 1 but later stubbed out in `pipeline.py` in favor of zero-maps. The core work is: (1) upgrade the beamforming modules with better DSP, (2) re-wire `process_chunk` to call the real engine, and (3) add the demand-driven activation gate.

**Primary recommendation:** Enhance the existing beamforming modules in-place (bandpass filter, parabolic interpolation, multi-peak, MCRA noise estimator), then replace the zero-map stub in `BeamformingPipeline.process_chunk` with a call to the real engine gated by CNN detection state.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BF-10 | Beamforming operates in 500-4000 Hz frequency band respecting UMA-16v2 spatial aliasing limit at ~4083 Hz | Bandpass filter section; spatial aliasing calc f=c/(2d)=343/(2*0.042)=4083 Hz |
| BF-11 | Bandpass filter (500-4000 Hz, 4th-order Butterworth) applied per-channel before beamforming | scipy.signal.butter + sosfilt pattern; apply before FFT in prepare_fft |
| BF-12 | Sub-grid parabolic interpolation refines peak DOA to sub-degree accuracy | Parabolic interpolation on 3-point neighborhood around SRP grid peak |
| BF-13 | Multi-peak detection identifies multiple simultaneous sources with configurable threshold and minimum separation | Peak finding with angular separation constraint on SRP map |
| BF-14 | MCRA noise estimator tracks adaptive noise floor for outdoor robustness | MCRA algorithm operating on SRP map power values |
| BF-15 | Beamforming wired into live pipeline's process_chunk (replacing current stub) | Replace zero-map stub in BeamformingPipeline.process_chunk |
| BF-16 | Beamforming activates only after CNN drone detection and deactivates after 5s of no detection | Demand-driven gate using state machine + monotonic timer |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Python backend, single Docker container
- Custom SRP-PHAT over Acoular (POC's 180-line implementation is simpler and sufficient for 4x4 array) -- locked decision
- Callback-based sounddevice.InputStream -- locked decision
- Audio capture callback does only np.copyto + monotonic timestamp -- no logging in audio thread
- Pipeline thread consumes chunks from ring buffer at 150ms intervals
- scipy>=1.14 already in requirements.txt
- numpy>=1.26,<3 already in requirements.txt

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | >=1.14 | Butterworth filter design + application | `scipy.signal.butter(output='sos')` + `sosfilt` for numerically stable real-time filtering |
| numpy | >=1.26,<3 | All beamforming math (FFT, cross-correlation, steering vectors) | Already the DSP core |

### No New Dependencies
This phase requires zero new packages. All algorithms (bandpass filter, parabolic interpolation, multi-peak detection, MCRA noise estimation) are implemented using scipy.signal and numpy, both already in requirements.txt.

## Architecture Patterns

### Current File Structure (beamforming/)
```
src/acoustic/beamforming/
    __init__.py          # Public API exports
    gcc_phat.py          # GCC-PHAT cross-correlation + FFT prep
    geometry.py          # UMA-16v2 mic positions + steering vectors
    peak.py              # Peak detection with adaptive threshold
    srp_phat.py          # 2D SRP-PHAT engine
```

### Target File Structure (additions)
```
src/acoustic/beamforming/
    __init__.py          # Updated exports
    bandpass.py          # NEW: Butterworth bandpass filter (BF-11)
    gcc_phat.py          # Unchanged
    geometry.py          # Unchanged
    interpolation.py     # NEW: Parabolic sub-grid interpolation (BF-12)
    mcra.py              # NEW: MCRA noise estimator (BF-14)
    multi_peak.py        # NEW: Multi-peak detection (BF-13)
    peak.py              # Updated: uses interpolation + multi-peak
    srp_phat.py          # Unchanged (filter applied before calling it)
```

### Pattern 1: Bandpass Pre-Filter (BF-11)
**What:** Apply 4th-order Butterworth bandpass (500-4000 Hz) per-channel before FFT
**When to use:** Every process_chunk call when beamforming is active
**Key detail:** Use second-order sections (SOS) format for numerical stability. Design filter once at init, apply per-chunk with `sosfilt`. For chunk-based processing, maintain filter state (`zi`) across chunks to avoid transient artifacts at chunk boundaries.

```python
# Source: scipy.signal.butter docs
from scipy.signal import butter, sosfilt, sosfilt_zi

class BandpassFilter:
    def __init__(self, fs: int, fmin: float = 500.0, fmax: float = 4000.0, order: int = 4):
        nyq = fs / 2.0
        low = fmin / nyq
        high = fmax / nyq
        self._sos = butter(order, [low, high], btype='band', output='sos')
        # Per-channel filter state for streaming
        self._zi: np.ndarray | None = None
        self._n_channels: int = 0

    def reset(self, n_channels: int) -> None:
        """Initialize filter state for n channels."""
        zi = sosfilt_zi(self._sos)  # shape (n_sections, 2)
        # Broadcast to (n_channels, n_sections, 2)
        self._zi = np.repeat(zi[np.newaxis, :, :], n_channels, axis=0)
        self._n_channels = n_channels

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Apply bandpass filter per-channel with state preservation.

        Args:
            signals: shape (n_mics, n_samples)
        Returns:
            Filtered signals, same shape
        """
        if self._zi is None or signals.shape[0] != self._n_channels:
            self.reset(signals.shape[0])
        filtered = np.empty_like(signals)
        for ch in range(signals.shape[0]):
            filtered[ch], self._zi[ch] = sosfilt(self._sos, signals[ch], zi=self._zi[ch])
        return filtered
```

### Pattern 2: Parabolic Sub-Grid Interpolation (BF-12)
**What:** Refine the grid-quantized peak location to sub-degree accuracy using 3-point parabolic fit
**When to use:** After finding the peak grid cell in the SRP map

The standard 1D parabolic interpolation formula for a peak at index `k` with neighbors:
```
delta = 0.5 * (y[k-1] - y[k+1]) / (y[k-1] - 2*y[k] + y[k+1])
refined_position = grid[k] + delta * grid_spacing
```

For 2D SRP maps, apply independently along azimuth and elevation axes:

```python
def parabolic_interpolation_2d(
    srp_map: np.ndarray,
    az_idx: int, el_idx: int,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
) -> tuple[float, float]:
    """Refine peak position via 2D parabolic interpolation.

    Returns refined (az_deg, el_deg) with sub-grid accuracy.
    """
    n_az, n_el = srp_map.shape

    # Azimuth refinement (if not at boundary)
    az_refined = float(az_grid_deg[az_idx])
    if 0 < az_idx < n_az - 1:
        y_l = srp_map[az_idx - 1, el_idx]
        y_c = srp_map[az_idx, el_idx]
        y_r = srp_map[az_idx + 1, el_idx]
        denom = y_l - 2 * y_c + y_r
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_l - y_r) / denom
            az_step = az_grid_deg[1] - az_grid_deg[0]
            az_refined += delta * az_step

    # Elevation refinement (if not at boundary)
    el_refined = float(el_grid_deg[el_idx])
    if 0 < el_idx < n_el - 1:
        y_l = srp_map[az_idx, el_idx - 1]
        y_c = srp_map[az_idx, el_idx]
        y_r = srp_map[az_idx, el_idx + 1]
        denom = y_l - 2 * y_c + y_r
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_l - y_r) / denom
            el_step = el_grid_deg[1] - el_grid_deg[0]
            el_refined += delta * el_step

    return az_refined, el_refined
```

### Pattern 3: MCRA Noise Estimator (BF-14)
**What:** Minima Controlled Recursive Averaging -- tracks noise floor adaptively by combining recursive averaging with minimum tracking
**When to use:** Updated every SRP-PHAT frame to maintain an adaptive threshold

The MCRA algorithm (Cohen & Berdugo, 2002) works on power spectral values. For beamforming, we adapt it to operate on the SRP map power:

1. **Smooth the power estimate:** `S(k) = alpha_s * S_prev(k) + (1 - alpha_s) * P(k)` where P(k) is the current SRP power at grid point k
2. **Track minimum over L frames:** `S_min(k) = min(S_min_prev(k), S(k))` with periodic reset every L frames
3. **Compute speech presence indicator:** `I(k) = 1 if S(k) / S_min(k) > delta` (delta is a threshold, typically 3-5)
4. **Compute conditional smoothing:** `alpha_d(k) = alpha_d_0 * I(k) + (1 - I(k)) * max(alpha_d_0, p_hat(k))` where p_hat is a smoothed presence probability
5. **Update noise estimate:** `noise(k) = alpha_d(k) * noise_prev(k) + (1 - alpha_d(k)) * P(k)`

Key parameters for outdoor drone detection:
- `alpha_s = 0.8` (power smoothing -- moderate to track changing noise)
- `alpha_d = 0.95` (noise update smoothing -- slow update during speech/signal presence)
- `L = 50` frames (~7.5s at 150ms chunks) for minimum tracking window
- `delta = 5.0` (signal presence threshold -- higher than speech default of 3 to avoid false triggers from wind)

```python
class MCRANoiseEstimator:
    """MCRA noise floor estimator adapted for SRP-PHAT maps."""

    def __init__(
        self,
        alpha_s: float = 0.8,
        alpha_d: float = 0.95,
        delta: float = 5.0,
        min_window: int = 50,
    ):
        self._alpha_s = alpha_s
        self._alpha_d = alpha_d
        self._delta = delta
        self._min_window = min_window
        self._frame_count = 0
        self._S: np.ndarray | None = None      # Smoothed power
        self._S_min: np.ndarray | None = None   # Tracked minimum
        self._noise: np.ndarray | None = None   # Noise estimate

    def update(self, srp_map: np.ndarray) -> np.ndarray:
        """Update noise estimate with new SRP map. Returns noise floor array."""
        power = srp_map.ravel()

        if self._S is None:
            # Initialize on first frame
            self._S = power.copy()
            self._S_min = power.copy()
            self._noise = power.copy()
            self._frame_count = 1
            return self._noise.reshape(srp_map.shape)

        # Step 1: Smooth power
        self._S = self._alpha_s * self._S + (1 - self._alpha_s) * power

        # Step 2: Track minimum
        self._S_min = np.minimum(self._S_min, self._S)
        self._frame_count += 1
        if self._frame_count % self._min_window == 0:
            self._S_min = self._S.copy()

        # Step 3: Signal presence indicator
        ratio = self._S / (self._S_min + 1e-10)
        signal_present = ratio > self._delta

        # Step 4: Conditional noise update
        alpha = np.where(signal_present, self._alpha_d, 0.5)
        self._noise = alpha * self._noise + (1 - alpha) * power

        return self._noise.reshape(srp_map.shape)

    def reset(self) -> None:
        """Reset estimator state."""
        self._S = None
        self._S_min = None
        self._noise = None
        self._frame_count = 0
```

### Pattern 4: Multi-Peak Detection (BF-13)
**What:** Find multiple peaks in the SRP map that exceed a threshold and are separated by a minimum angular distance
**When to use:** After computing the SRP map, before reporting detections

```python
def detect_multi_peak(
    srp_map: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    noise_floor: np.ndarray,
    threshold_factor: float = 3.0,
    min_separation_deg: float = 15.0,
    max_peaks: int = 5,
) -> list[PeakDetection]:
    """Detect multiple peaks exceeding noise floor * threshold with minimum separation."""
    threshold = noise_floor * threshold_factor
    candidates = np.argwhere(srp_map > threshold)
    # Sort by power descending
    powers = [srp_map[tuple(c)] for c in candidates]
    order = np.argsort(powers)[::-1]
    candidates = candidates[order]

    peaks: list[PeakDetection] = []
    for c in candidates:
        if len(peaks) >= max_peaks:
            break
        az = az_grid_deg[c[0]]
        el = el_grid_deg[c[1]]
        # Check angular separation from all accepted peaks
        too_close = False
        for p in peaks:
            dist = np.sqrt((az - p.az_deg)**2 + (el - p.el_deg)**2)
            if dist < min_separation_deg:
                too_close = True
                break
        if not too_close:
            peaks.append(PeakDetection(
                az_deg=float(az), el_deg=float(el),
                power=float(srp_map[c[0], c[1]]),
                threshold=float(threshold[c[0], c[1]]),
            ))
    return peaks
```

### Pattern 5: Demand-Driven Beamforming Gate (BF-16)
**What:** Only run SRP-PHAT when CNN has detected a drone; return zero-map otherwise
**When to use:** In `process_chunk`, check CNN state before running expensive beamforming

```python
# In BeamformingPipeline:
def process_chunk(self, chunk: np.ndarray) -> list[PeakDetection]:
    now = time.monotonic()

    # Check if CNN has active detection
    cnn_active = (
        self._state_machine is not None
        and self._state_machine.state == DetectionState.CONFIRMED
    )

    if cnn_active:
        self._last_bf_active_time = now

    # Keep beamforming active for 5s after last CNN detection
    bf_should_run = (now - self._last_bf_active_time) < self._bf_holdoff

    if not bf_should_run:
        self.latest_map = np.zeros((self._az_size, self._el_size), dtype=np.float32)
        self.latest_peak = None
        self._last_process_time = now
        return []

    # Run real beamforming
    signals = chunk.T  # (n_mics, n_samples)
    filtered = self._bandpass.apply(signals)
    srp_map = srp_phat_2d(filtered, self._mic_positions, ...)
    # ... peak detection, interpolation, etc.
```

### Anti-Patterns to Avoid
- **Applying bandpass via frequency masking in GCC-PHAT only:** The existing `prepare_fft` already does frequency masking, but a proper time-domain Butterworth filter gives sharper rolloff and avoids spectral leakage. Apply the bandpass BEFORE FFT, keep the GCC-PHAT band_mask as a secondary safeguard.
- **Re-creating filter each chunk:** Design the Butterworth filter once at init. Only `sosfilt` is called per chunk.
- **Resetting filter state on every chunk:** Maintain `zi` state across chunks for seamless streaming. Only reset on pipeline restart or device reconnect.
- **Running beamforming unconditionally:** The SRP-PHAT loop over 120 mic pairs is CPU-intensive. Gate behind CNN detection to save compute when no drone is present.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Butterworth filter design | Manual IIR coefficient calculation | `scipy.signal.butter(output='sos')` | Numerically stable SOS form, handles edge cases |
| Streaming filter application | Manual difference equation | `scipy.signal.sosfilt` with `zi` state | Correct state management across chunks |
| Filter initial conditions | Zero-initialize zi | `scipy.signal.sosfilt_zi` | Avoids startup transient |

**Key insight:** The bandpass filter and its streaming state management are the only parts that benefit from scipy. MCRA, parabolic interpolation, and multi-peak detection are simple enough to implement in pure numpy and don't have off-the-shelf equivalents that match this use case.

## Common Pitfalls

### Pitfall 1: Filter State Reset on Chunk Boundaries
**What goes wrong:** Bandpass filter produces clicking/artifacts at every 150ms chunk boundary
**Why it happens:** Filter state (`zi`) not preserved between calls to `sosfilt`
**How to avoid:** Store `zi` per channel, pass it to `sosfilt` and capture the returned state
**Warning signs:** Audible artifacts at 6.67 Hz (1/0.15s) in filtered output

### Pitfall 2: Spatial Aliasing Above 4083 Hz
**What goes wrong:** Beamforming produces ghost peaks (aliased directions) for high-frequency content
**Why it happens:** UMA-16v2 spacing (42mm) sets aliasing limit at f = c/(2d) = 343/(2*0.042) = 4083 Hz
**How to avoid:** Bandpass upper limit at 4000 Hz (below aliasing limit) enforced by BF-10
**Warning signs:** Spurious peaks appearing in unexpected directions

### Pitfall 3: MCRA Initialization Transient
**What goes wrong:** First few seconds after startup produce either no detections or false detections
**Why it happens:** MCRA needs ~L frames (50 frames = 7.5s) to build a stable noise estimate
**How to avoid:** Initialize noise estimate from first frame, use conservative threshold during warmup period
**Warning signs:** Inconsistent detection behavior in first 5-10 seconds after startup

### Pitfall 4: Parabolic Interpolation at Grid Boundaries
**What goes wrong:** Index out of bounds when peak is at the edge of the azimuth or elevation grid
**Why it happens:** Parabolic fit requires neighbors on both sides (k-1 and k+1)
**How to avoid:** Check boundary conditions -- if peak is at grid edge, return grid-quantized value without interpolation
**Warning signs:** IndexError or ArrayOutOfBounds exceptions

### Pitfall 5: Demand-Driven Gate Race Condition
**What goes wrong:** Beamforming never activates because CNN result arrives after `process_chunk` checks state
**Why it happens:** CNN inference runs asynchronously in a separate worker thread; state machine update happens in `_process_cnn` which runs after `process_chunk`
**How to avoid:** Check the CNN state at the START of the pipeline loop iteration, not inside `process_chunk`. Or: check state machine state directly, which was updated in the previous iteration's `_process_cnn` call.
**Warning signs:** CNN shows "CONFIRMED" state but beamforming map stays zero

### Pitfall 6: Diagonal Spacing and Effective Aliasing
**What goes wrong:** Aliasing appears below 4083 Hz for diagonal source directions
**Why it happens:** Diagonal mic spacing is 42*sqrt(2) = 59.4mm, giving aliasing at ~2888 Hz for diagonal directions
**How to avoid:** The 4000 Hz upper limit from BF-10 is safe for broadside but marginal for diagonal. Accept this as a known limitation of the 4x4 planar array -- document but don't try to fix. The requirement explicitly specifies 4083 Hz.
**Warning signs:** Ghost peaks for sources at 45-degree angles when high-frequency content (>3000 Hz) is present

## Code Examples

### Spatial Aliasing Limit Calculation
```python
# UMA-16v2 spatial aliasing
SPACING = 0.042  # meters (42mm adjacent)
C = 343.0  # speed of sound m/s
f_alias = C / (2 * SPACING)  # = 4083 Hz
# BF-10 requires fmax = 4000 Hz (below aliasing limit)
```

### Config Extensions Needed
```python
# In AcousticSettings (config.py):

# Beamforming frequency band (BF-10, override existing freq_min/freq_max)
bf_freq_min: float = 500.0
bf_freq_max: float = 4000.0
bf_filter_order: int = 4

# Multi-peak detection (BF-13)
bf_min_separation_deg: float = 15.0
bf_max_peaks: int = 5
bf_peak_threshold: float = 3.0  # factor above noise floor

# MCRA noise estimation (BF-14)
bf_mcra_alpha_s: float = 0.8
bf_mcra_alpha_d: float = 0.95
bf_mcra_delta: float = 5.0
bf_mcra_min_window: int = 50

# Demand-driven activation (BF-16)
bf_holdoff_seconds: float = 5.0
```

### Pipeline Integration Pattern (BF-15)
```python
# In pipeline.py __init__:
from acoustic.beamforming import (
    build_mic_positions, srp_phat_2d,
)
from acoustic.beamforming.bandpass import BandpassFilter
from acoustic.beamforming.mcra import MCRANoiseEstimator
from acoustic.beamforming.multi_peak import detect_multi_peak
from acoustic.beamforming.interpolation import parabolic_interpolation_2d

self._mic_positions = build_mic_positions()
self._bandpass = BandpassFilter(settings.sample_rate, settings.bf_freq_min, settings.bf_freq_max, settings.bf_filter_order)
self._mcra = MCRANoiseEstimator(...)
self._az_grid = np.arange(-settings.az_range, settings.az_range + settings.az_resolution, settings.az_resolution)
self._el_grid = np.arange(-settings.el_range, settings.el_range + settings.el_resolution, settings.el_resolution)
self._last_bf_active_time = 0.0  # monotonic
self._bf_holdoff = settings.bf_holdoff_seconds
```

## State of the Art

| Old Approach (Phase 1) | Current Approach (Phase 17) | Why Changed |
|------------------------|----------------------------|-------------|
| freq_min=100, freq_max=2000 Hz | 500-4000 Hz | Research-validated drone band; old band missed rotor harmonics above 2kHz |
| Frequency masking in FFT domain only | Butterworth bandpass pre-filter + FFT masking | Sharper rolloff, reduces noise outside band |
| Single peak via argmax | Multi-peak with MCRA noise floor | Supports multiple drones simultaneously |
| Grid-quantized DOA (1-degree steps) | Sub-grid parabolic interpolation | Sub-degree accuracy for smoother bearing |
| Percentile-based noise threshold | MCRA adaptive noise estimator | Adapts to changing outdoor conditions without manual recalibration |
| Always-on beamforming | Demand-driven (CNN-gated) | Saves compute; SRP-PHAT loop over 120 pairs is expensive |
| Beamforming stubbed out (zero-map) | Wired into process_chunk | Restores real functionality |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BF-10 | Frequency band 500-4000 Hz, aliasing respected | unit | `pytest tests/unit/test_bandpass.py -x` | No -- Wave 0 |
| BF-11 | 4th-order Butterworth bandpass per-channel | unit | `pytest tests/unit/test_bandpass.py -x` | No -- Wave 0 |
| BF-12 | Parabolic interpolation sub-degree accuracy | unit | `pytest tests/unit/test_interpolation.py -x` | No -- Wave 0 |
| BF-13 | Multi-peak detection with separation | unit | `pytest tests/unit/test_multi_peak.py -x` | No -- Wave 0 |
| BF-14 | MCRA noise estimator tracking | unit | `pytest tests/unit/test_mcra.py -x` | No -- Wave 0 |
| BF-15 | Pipeline calls real beamforming engine | integration | `pytest tests/integration/test_pipeline.py -x` | Yes -- needs update |
| BF-16 | Demand-driven activation/deactivation | unit | `pytest tests/unit/test_bf_gate.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x -q`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_bandpass.py` -- covers BF-10, BF-11
- [ ] `tests/unit/test_interpolation.py` -- covers BF-12
- [ ] `tests/unit/test_multi_peak.py` -- covers BF-13
- [ ] `tests/unit/test_mcra.py` -- covers BF-14
- [ ] `tests/unit/test_bf_gate.py` -- covers BF-16
- [ ] `tests/integration/test_pipeline.py` -- update existing for BF-15

## Open Questions

1. **Existing freq_min/freq_max config vs new bf_freq_min/bf_freq_max**
   - What we know: Current config has `freq_min=100.0` and `freq_max=2000.0` used by `srp_phat_2d` fmin/fmax params
   - What's unclear: Should we change the existing fields or add new bf-prefixed ones?
   - Recommendation: Add new `bf_freq_min` and `bf_freq_max` fields for the upgraded beamforming, keep old ones for backward compat. The bandpass filter uses the new fields; the GCC-PHAT band_mask also uses the new fields.

2. **PeakDetection type for multi-peak**
   - What we know: Current `PeakDetection` is a single dataclass; `latest_peak` stores one peak
   - What's unclear: How to propagate multiple peaks downstream
   - Recommendation: Change `latest_peak` to `latest_peaks: list[PeakDetection]` and keep `latest_peak` as a property returning the first (strongest) peak for backward compatibility. Phase 18 (DOA) will consume the list.

3. **MCRA warm-up period behavior**
   - What we know: MCRA needs ~50 frames (~7.5s) to stabilize
   - What's unclear: Should beamforming produce results during warm-up?
   - Recommendation: Yes -- use the percentile-based threshold from the existing `detect_peak_with_threshold` as fallback during MCRA warm-up (first 50 frames), then switch to MCRA-based threshold.

## Sources

### Primary (HIGH confidence)
- [scipy.signal.butter docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) -- filter design, SOS format
- [scipy.signal.sosfilt docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html) -- streaming filter application
- [scipy Butterworth Bandpass Cookbook](https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html) -- practical examples
- Existing codebase: `src/acoustic/beamforming/` -- verified current implementation
- Existing codebase: `src/acoustic/pipeline.py` -- verified stub state and _process_cnn flow

### Secondary (MEDIUM confidence)
- [SRP-PHAT Tutorial Review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11557718/) -- parabolic interpolation, multi-source detection, computational guidelines
- [Cohen & Berdugo MCRA Paper](https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf) -- original MCRA algorithm (IEEE Signal Processing Letters, 2002)
- [SRP-PHAT Wikipedia](https://en.wikipedia.org/wiki/Steered-Response_Power_Phase_Transform) -- general reference
- [UMA-16v2 Product Brief](https://www.minidsp.com/images/documents/Product%20Brief-UMA16%20v2.pdf) -- array specifications

### Tertiary (LOW confidence)
- [Interpolation Methods for SRP-PHAT (Academia)](https://www.academia.edu/2737556/Interpolation_methods_for_the_SRP-PHAT_algorithm) -- comparative study of interpolation methods (not verified in detail)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- scipy and numpy already in project, no new deps needed
- Architecture: HIGH -- existing beamforming modules are well-understood, enhancement paths are clear
- Pitfalls: HIGH -- filter state management and spatial aliasing are well-documented concerns
- MCRA algorithm: MEDIUM -- adapted from speech processing literature to SRP-PHAT power maps; parameters may need tuning for outdoor drone detection
- Parabolic interpolation: HIGH -- standard DSP technique with simple formula

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable domain -- DSP algorithms don't change)
