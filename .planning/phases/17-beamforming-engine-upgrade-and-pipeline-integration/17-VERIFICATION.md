---
phase: 17-beamforming-engine-upgrade-and-pipeline-integration
verified: 2026-04-06T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
gaps: []
human_verification:
  - test: "Run the live system with UMA-16v2 connected, play a drone audio recording, and observe beamforming activates only after CNN confirms a drone"
    expected: "Beamforming gate opens on DRONE_CONFIRMED, holds for 5 seconds after drone stops, then idles"
    why_human: "Requires hardware (UMA-16v2), real-time CNN inference, and live audio — cannot verify programmatically"
  - test: "Observe the beamforming spatial map in the web UI while running with a known sound source at a fixed direction"
    expected: "SRP-PHAT map shows a clear peak in the correct angular direction, refined by parabolic interpolation to sub-degree precision"
    why_human: "Requires hardware and visual inspection of the live map — spatial accuracy of the full stack cannot be verified without real audio"
---

# Phase 17: Beamforming Engine Upgrade and Pipeline Integration — Verification Report

**Phase Goal:** Upgrade beamforming engine (bandpass pre-filter, parabolic interpolation, MCRA noise, multi-peak detection) and wire into live pipeline with demand-driven activation.
**Verified:** 2026-04-06
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Audio is filtered to 500-4000 Hz before beamforming, respecting UMA-16v2 spatial aliasing limit | VERIFIED | `BandpassFilter` in `bandpass.py` uses `butter(order, [low, high], btype='band', output='sos')` with defaults `fmin=500.0, fmax=4000.0`. Called in `pipeline.py` line 181 via `self._bandpass.apply(signals)`. 9 unit tests all pass. |
| 2 | Peak DOA is refined to sub-degree accuracy via parabolic interpolation | VERIFIED | `parabolic_interpolation_2d` in `interpolation.py` implements parabolic fitting with boundary handling. Called in `pipeline.py` lines 214-216 for every detected peak. 5 unit tests all pass including analytical delta=1/6 degree case. |
| 3 | Config fields for ALL beamforming parameters exist and default correctly | VERIFIED | `config.py` contains all 12 `bf_*` fields: `bf_freq_min=500.0`, `bf_freq_max=4000.0`, `bf_filter_order=4`, `bf_min_separation_deg=15.0`, `bf_max_peaks=5`, `bf_peak_threshold=3.0`, `bf_mcra_alpha_s=0.8`, `bf_mcra_alpha_d=0.95`, `bf_mcra_delta=5.0`, `bf_mcra_min_window=50`, `bf_holdoff_seconds=5.0`. Verified by test `test_config_all_bf_fields`. |
| 4 | MCRA noise estimator adapts its noise floor estimate over time without manual recalibration | VERIFIED | `MCRANoiseEstimator` in `mcra.py` uses smoothed minimum tracking with secondary global-median signal presence detection. 7 unit tests all pass including convergence and signal preservation tests. |
| 5 | Multiple simultaneous sources are detected as separate peaks with configurable separation | VERIFIED | `detect_multi_peak` in `multi_peak.py` implements greedy angular separation algorithm. Returns `list[PeakDetection]` sorted by power descending. 6 unit tests all pass including separation constraint and max_peaks limit. |
| 6 | Pipeline `process_chunk` calls real SRP-PHAT beamforming instead of returning zero maps | VERIFIED | `pipeline.py` lines 184-193 call `srp_phat_2d(filtered, self._mic_positions, ...)`. No stub language remains. Manual spot-check confirms `latest_map` is non-zero after processing a synthetic chunk. |
| 7 | Beamforming activates only when CNN detects a drone and stays active for 5 seconds after last detection | VERIFIED | `pipeline.py` lines 158-175 implement demand-driven gate using `_last_bf_active_time` and `bf_holdoff_seconds`. 7 unit tests in `test_bf_gate.py` all pass covering gate-on, gate-off, holdoff within window (4.9s), holdoff expired (5.1s), and no-state-machine fallback. |
| 8 | When no drone is detected and holdoff has expired, `process_chunk` returns zero map without running SRP-PHAT | VERIFIED | `pipeline.py` lines 168-175: `if not bf_should_run` returns `[]` and sets `latest_map` to zeros. Test `test_returns_empty_list_when_no_drone_and_holdoff_expired` passes. |
| 9 | Multiple peaks are detected and the strongest is used for CNN bearing | VERIFIED | `pipeline.py` line 342: `best = peaks[0]` (peaks sorted by power descending). `latest_peaks` stores all peaks; `latest_peak` stores first. `_process_cnn` uses `peaks[0]` for bearing push to CNN worker. |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/acoustic/beamforming/bandpass.py` | BandpassFilter class with streaming per-channel Butterworth filter | VERIFIED | 70 lines. Exports `BandpassFilter`. `butter`, `sosfilt`, `sosfilt_zi` imported from `scipy.signal`. `apply()` and `reset()` implemented. |
| `src/acoustic/beamforming/interpolation.py` | Parabolic sub-grid interpolation for 2D SRP maps | VERIFIED | 69 lines. Exports `parabolic_interpolation_2d`. Handles az and el axes independently with boundary guard. |
| `src/acoustic/beamforming/mcra.py` | MCRANoiseEstimator class with update/reset interface | VERIFIED | 106 lines. Exports `MCRANoiseEstimator`. `update()` and `reset()` implemented. Global-median secondary signal presence detection included. |
| `src/acoustic/beamforming/multi_peak.py` | detect_multi_peak function returning list[PeakDetection] | VERIFIED | 90 lines. Exports `detect_multi_peak`. Imports `PeakDetection` from `acoustic.types`. Greedy angular separation algorithm implemented. |
| `src/acoustic/config.py` | All bf_* config fields with correct defaults | VERIFIED | Contains all 12 `bf_*` fields. Duplicate block removed by Plan 03 fix. Legacy `freq_min/freq_max` preserved for backward compat. |
| `src/acoustic/pipeline.py` | Real beamforming in process_chunk with demand-driven gate | VERIFIED | 449 lines. Contains `srp_phat_2d`, `self._bandpass.apply`, `self._mcra.update`, `detect_multi_peak`, `parabolic_interpolation_2d`, `bf_should_run`, `self._last_bf_active_time`, `self.latest_peaks`. No stub language remains. |
| `src/acoustic/beamforming/__init__.py` | Updated exports including new modules | VERIFIED | Exports `BandpassFilter`, `MCRANoiseEstimator`, `detect_multi_peak`, `parabolic_interpolation_2d` plus all prior exports. All four in `__all__`. |
| `tests/unit/test_bandpass.py` | Unit tests for bandpass filter | VERIFIED | 9 tests all pass. |
| `tests/unit/test_interpolation.py` | Unit tests for parabolic interpolation | VERIFIED | 5 tests all pass. |
| `tests/unit/test_mcra.py` | Unit tests for MCRA noise estimator | VERIFIED | 7 tests all pass. |
| `tests/unit/test_multi_peak.py` | Unit tests for multi-peak detection | VERIFIED | 6 tests all pass. |
| `tests/unit/test_bf_gate.py` | Unit tests for demand-driven beamforming gate | VERIFIED | 7 tests all pass. |
| `tests/integration/test_pipeline_beamforming.py` | Integration test for full pipeline with real beamforming | VERIFIED (partial) | File exists with 5 tests covering real map, gate-blocks, gate-allows, and CNN integration. Tests cannot be run via pytest due to a pre-existing FastAPI import error in `tests/integration/conftest.py` (unrelated to phase 17 — `static.py` `Union` return type incompatible with current FastAPI version). Core logic verified directly via Python import checks (all assertions pass). |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pipeline.py` | `bandpass.py` | `self._bandpass.apply(signals)` | WIRED | Line 181 — called on every active beamforming chunk |
| `pipeline.py` | `srp_phat.py` | `srp_phat_2d(filtered, ...)` | WIRED | Lines 184-193 — called with filtered signals and mic positions |
| `pipeline.py` | `mcra.py` | `self._mcra.update(srp_map)` | WIRED | Line 197 — MCRA receives SRP map, returns noise floor |
| `pipeline.py` | `multi_peak.py` | `detect_multi_peak(srp_map, ...)` | WIRED | Lines 200-208 — called with noise floor from MCRA |
| `pipeline.py` | `interpolation.py` | `parabolic_interpolation_2d(...)` | WIRED | Lines 214-216 — called per-peak to refine az/el |
| `multi_peak.py` | `types.py` | `from acoustic.types import PeakDetection` | WIRED | Line 14 — returns `list[PeakDetection]` |
| `mcra.py` | `srp_phat output` | `update(self, srp_map: np.ndarray) -> np.ndarray` | WIRED | Takes SRP map, returns noise floor array of same shape |
| `beamforming/__init__.py` | new modules | re-exports BandpassFilter, MCRANoiseEstimator, detect_multi_peak, parabolic_interpolation_2d | WIRED | All four in `__all__` and importable from `acoustic.beamforming` |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `pipeline.py` process_chunk | `srp_map` | `srp_phat_2d(filtered, mic_positions, ...)` | Yes — real FFT-based SRP computation on bandpass-filtered audio | FLOWING |
| `pipeline.py` process_chunk | `noise_floor` | `self._mcra.update(srp_map)` | Yes — adaptive MCRA estimate, initialized from real SRP map | FLOWING |
| `pipeline.py` process_chunk | `peaks` | `detect_multi_peak(srp_map, az_grid, el_grid, noise_floor, ...)` | Yes — thresholded against real noise floor | FLOWING |
| `pipeline.py` process_chunk | `filtered` | `self._bandpass.apply(signals)` | Yes — streaming Butterworth applied to live mic channels | FLOWING |
| `pipeline.py` latest_map | `srp_map.astype(np.float32)` | Stored from real SRP-PHAT output | Yes — non-zero verified by spot-check | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Verification | Result | Status |
|----------|-------------|--------|--------|
| Pipeline with no state machine produces non-zero SRP map | `pipe.process_chunk(chunk)` then `not np.all(pipe.latest_map == 0)` | map shape (181, 91), not all zeros | PASS |
| Gate blocks when NO_DRONE + holdoff expired | `pipe._last_bf_active_time = 0.0`, check `result == []` and `np.all(pipe.latest_map == 0)` | PASS | PASS |
| Gate allows when DRONE_CONFIRMED | FakeStateMachine('DRONE_CONFIRMED'), check map not all zeros | PASS | PASS |
| No state machine: beamforming always runs | BeamformingPipeline with no state_machine, check map not all zeros | PASS | PASS |
| 34 unit tests across all phase 17 modules | `python -m pytest tests/unit/test_bandpass.py test_interpolation.py test_mcra.py test_multi_peak.py test_bf_gate.py` | 34 passed in 35.42s | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|------------|-------------|--------|----------|
| BF-10 | 17-01 | Beamforming operates in 500-4000 Hz band respecting UMA-16v2 spatial aliasing limit | SATISFIED | `bf_freq_min=500.0`, `bf_freq_max=4000.0` in config; `BandpassFilter` enforces these bounds; `srp_phat_2d` called with `fmin/fmax` from config |
| BF-11 | 17-01 | Bandpass filter (4th-order Butterworth) applied per-channel before beamforming | SATISFIED | `BandpassFilter` uses `butter(order, ..., output='sos')` with `bf_filter_order=4`; called in `pipeline.py` before `srp_phat_2d` |
| BF-12 | 17-01 | Sub-grid parabolic interpolation refines peak DOA to sub-degree accuracy | SATISFIED | `parabolic_interpolation_2d` applied per-peak in `pipeline.py`; 5 unit tests pass including analytical verification |
| BF-13 | 17-02 | Multi-peak detection with configurable threshold and minimum separation | SATISFIED | `detect_multi_peak` with `min_separation_deg` and `max_peaks` params; wired into pipeline; 6 unit tests pass |
| BF-14 | 17-02 | MCRA noise estimator tracks adaptive noise floor | SATISFIED | `MCRANoiseEstimator` with alpha/delta/min_window params; wired into pipeline; 7 unit tests pass including convergence test |
| BF-15 | 17-03 | Beamforming wired into live pipeline's process_chunk (replacing stub) | SATISFIED | `pipeline.py` calls `srp_phat_2d`; no stub language remains; spot-check confirms non-zero map |
| BF-16 | 17-03 | Demand-driven activation after CNN detection, 5s holdoff | SATISFIED | `bf_should_run` gate in `process_chunk`; `_last_bf_active_time` updated on CONFIRMED; `bf_holdoff_seconds=5.0` from config; 7 gate unit tests pass |

**Note:** REQUIREMENTS.md still shows BF-10 through BF-16 as `[ ] Pending` — documentation not updated after phase completion. This is a documentation gap only; all implementations are present and verified.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `.planning/REQUIREMENTS.md` | 72-78 | BF-10 through BF-16 still marked `[ ]` (unchecked) and `Pending` in tracker table | Info | Documentation drift — no code impact. Should be updated to `[x]` and `Complete` status. |

No code anti-patterns found. No TODO/FIXME/placeholder comments in any phase 17 source files. No stub return values. All implementations contain real algorithms.

---

### Human Verification Required

**1. Demand-driven gate with live hardware**

**Test:** Connect UMA-16v2, start the service, play drone audio from a speaker at a fixed location, then stop. Observe the pipeline tab and WebSocket events.
**Expected:** Beamforming activates within 0.5 seconds of CNN confirming drone, runs for exactly 5 seconds after drone audio stops (bf_holdoff_seconds), then idles with zero map.
**Why human:** Requires UMA-16v2 hardware, real CNN inference model loaded, and real-time audio environment.

**2. Spatial accuracy of SRP-PHAT with parabolic refinement**

**Test:** Place a speaker at a known azimuth (e.g., 30 degrees) from the array, play a 2 kHz tone, observe `latest_peak.az_deg` reported by the pipeline.
**Expected:** Peak azimuth is within ±5 degrees of actual speaker position; parabolic refinement produces non-integer degree value (sub-grid accuracy confirmed).
**Why human:** Spatial accuracy depends on physical mic spacing calibration, room acoustics, and the actual array geometry — cannot be verified without hardware.

---

### Gaps Summary

No gaps. All phase 17 must-haves are verified at all levels (exists, substantive, wired, data-flowing).

**Integration test note:** `tests/integration/test_pipeline_beamforming.py` cannot be collected via pytest due to a pre-existing `FastAPIError` in `tests/integration/conftest.py` (`static.py` return type annotation incompatible with installed FastAPI version). This is a pre-existing issue unrelated to phase 17 — the same error blocks all integration tests. The phase 17 integration test logic was verified directly via Python imports and assertion checks; all 5 test behaviors pass. The conftest issue should be fixed in a separate task.

---

_Verified: 2026-04-06_
_Verifier: Claude (gsd-verifier)_
