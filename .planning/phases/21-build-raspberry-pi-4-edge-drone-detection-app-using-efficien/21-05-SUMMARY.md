---
phase: 21
plan: 05
subsystem: apps/rpi-edge
tags: [audio-capture, resample, onnxruntime, hysteresis, detection-engine, wave-2]
dependency_graph:
  requires:
    - 21-01  # Wave 0 RED stubs (test_resample_48_to_32, test_hysteresis, test_inference_latency_host)
    - 21-02  # NumpyMelSTFT (feeds the inference latency test)
    - 21-03  # Committed FP32 + int8 ONNX artifacts + sha256 manifest
    - 21-04  # ModelConfig dataclass consumed by OnnxClassifier
  provides:
    - apps/rpi-edge/skyfort_edge/audio.py (AudioCapture + resample_48k_to_32k)
    - apps/rpi-edge/skyfort_edge/hysteresis.py (HysteresisStateMachine + StateEvent)
    - apps/rpi-edge/skyfort_edge/inference.py (OnnxClassifier with checksum verify)
  affects:
    - apps/rpi-edge/tests/test_resample_48_to_32.py (RED -> GREEN)
    - apps/rpi-edge/tests/test_hysteresis.py (RED -> GREEN)
    - apps/rpi-edge/tests/test_inference_latency_host.py (RED -> GREEN)
    - Plans 21-06 / 21-07 (GPIO, detection log, HTTP, __main__ wire these modules)
tech_stack:
  added: []  # All deps pinned in 21-01 (scipy, onnxruntime, numpy)
  patterns:
    - Non-allocating PortAudio callback (T-21-13): only np.copyto + int++ under a lock
    - Pre-allocated float32 ring buffer, most-recent-window read
    - scipy.signal.resample_poly(up=2, down=3) 48->32 kHz polyphase resample
    - ONNX checksum gate (T-21-05) before ORT InferenceSession construction
    - Int8-preferred load with silent FP32 fallback; hard RuntimeError on total failure
    - K-of-N hysteresis with min_on_seconds hold, ported (not imported) from
      src/acoustic/classification/state_machine.py (D-12/D-25)
key_files:
  created:
    - apps/rpi-edge/skyfort_edge/audio.py
    - apps/rpi-edge/skyfort_edge/hysteresis.py
    - apps/rpi-edge/skyfort_edge/inference.py
  modified:
    - apps/rpi-edge/tests/test_resample_48_to_32.py
    - apps/rpi-edge/tests/test_hysteresis.py
    - apps/rpi-edge/tests/test_inference_latency_host.py
decisions:
  - "D-02: 48->32 kHz resampling uses scipy.signal.resample_poly(up=2, down=3). Per-call latency measured at well under 50 ms on host."
  - "D-05/D-07: int8 ONNX is the default; FP32 ONNX is a silent fallback. Missing primary or checksum mismatch logs a warning and falls through; total failure raises RuntimeError so the service never silently serves nothing."
  - "D-12/D-25: HysteresisStateMachine is a pure port of the service's state_machine.py semantics. No import from src.acoustic — grep-verified."
  - "T-21-05: Checksum gate uses models/efficientat_mn10_v6_onnx.sha256. Missing manifest logs a warning and permits load (dev workflow). Mismatch refuses load."
  - "T-21-13: AudioCapture._callback does only np.copyto + integer stat increments under a threading.Lock. No logging, no allocation, no string formatting on the PortAudio thread."
  - "Binary head reality (from 21-01 MODEL_HEAD_NOTES.md): classify() returns a length-1 logits vector and the caller applies sigmoid to get drone probability. num_classes=1."
metrics:
  duration_minutes: 20
  tasks_completed: 2
  tests_added: 11  # 2 resample + 5 hysteresis + 4 inference
  files_created: 3
  files_modified: 3
  completed_date: 2026-04-08
---

# Phase 21 Plan 05: Detection Engine Core (Audio + Inference + Hysteresis) Summary

Wires the edge app's data plane: single USB mic capture at 48 kHz, polyphase resample to 32 kHz, feed NumpyMelSTFT (21-02) into an int8-preferred ONNX Runtime session (21-03) with a sha256 tamper check, and funnel the per-window scores through a K-of-N hysteresis state machine. Three Wave 0 RED stubs (`test_resample_48_to_32`, `test_hysteresis`, `test_inference_latency_host`) are now GREEN and Plans 21-06 / 21-07 can wire these modules into GPIO, detection log, HTTP, and `__main__`.

## What Was Built

### Task 1 — AudioCapture + resample + HysteresisStateMachine (commit `66ccb76`)

**`apps/rpi-edge/skyfort_edge/audio.py`** — single-channel 48 kHz capture with a pre-allocated float32 ring buffer and a zero-allocation `_callback` (T-21-13 mitigation). Constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CAPTURE_SR` | 48000 | USB mic default |
| `INFERENCE_SR` | 32000 | Training mel filterbank SR |
| `RESAMPLE_UP` / `RESAMPLE_DOWN` | 2 / 3 | `48000 * 2 / 3 == 32000` |

Public API:

- `AudioCapture(device=None, ring_seconds=4.0)`: constructs the ring buffer.
- `.start()` / `.stop()`: opens a 50 ms-block `sounddevice.InputStream` (callback-based, PortAudio).
- `.read_window_48k(duration_seconds)`: most-recent slice of the ring at 48 kHz.
- `.read_window_32k(duration_seconds)`: same, then resampled to 32 kHz via `resample_48k_to_32k`.
- `resample_48k_to_32k(x)` module helper: `scipy.signal.resample_poly(x, up=2, down=3).astype(float32)`. Used standalone by tests and by `AudioCapture.read_window_32k`.

The `_callback` path performs only `np.copyto` into the ring slice under a `threading.Lock` plus two integer increments; no `log.*`, no f-strings, no allocation. Wrap for `time_info.inputBufferAdcTime` is `try/except` so malformed PortAudio timestamps cannot raise into the audio thread.

**`apps/rpi-edge/skyfort_edge/hysteresis.py`** — `HysteresisStateMachine` ported (not imported — D-25 grep-verified) from `src/acoustic/classification/state_machine.py`. Enum `State` (`IDLE`, `LATCHED`), enum `EventType` (`RISING_EDGE`, `FALLING_EDGE`), dataclass `StateEvent(type, timestamp, score, latch_duration_seconds)`.

Constructor guards reject `exit_threshold > enter_threshold`, `confirm_hits < 1`, `release_hits < 1`, and `min_on_seconds < 0`. `update(score, timestamp)` advances the machine and returns a `StateEvent` only on actual edge transitions.

Latch release requires BOTH `release_hits` consecutive below-exit hits AND `timestamp - latch_start_time >= min_on_seconds` AND `timestamp - last_positive_time >= min_on_seconds`. This is a port of CLS-03 semantics, not a reinvention.

**`apps/rpi-edge/tests/test_resample_48_to_32.py`** — 2 GREEN tests:
- `test_resample_poly_2_3_correctness`: length check (48000 → 32000), dtype preservation, silence preservation, and a 1 kHz sinusoid round-trip that verifies the FFT peak bin after resampling lands within 2 Hz of 1000 Hz.
- `test_resample_latency_under_50ms_for_1s_window`: 10-iteration average latency must be < 50 ms.

**`apps/rpi-edge/tests/test_hysteresis.py`** — 5 GREEN tests:
- `test_rising_edge_latches_after_confirm_hits`: exactly `confirm_hits=3` consecutive above-hits latches.
- `test_below_enter_resets_confirm_counter`: a mid-stream sub-enter score resets the counter.
- `test_min_on_seconds_held_after_last_positive`: scores drop immediately but state stays LATCHED through `min_on_seconds`.
- `test_release_after_release_hits_below_exit_threshold`: after the min-on window, `release_hits=5` below-exit scores trip the falling edge and report `latch_duration_seconds > 2.0`.
- `test_exit_threshold_must_be_below_enter_threshold`: constructor raises `ValueError` on inverted thresholds.

Result: `pytest tests/test_resample_48_to_32.py tests/test_hysteresis.py` → **7 passed**.

### Task 2 — OnnxClassifier + latency test (commit `9d095e8`)

**`apps/rpi-edge/skyfort_edge/inference.py`** — `class OnnxClassifier`:

- `__init__(model_cfg)` walks `[primary, fallback]` candidates where primary is int8 when `prefer_int8=True` (default) and FP32 otherwise. For each candidate:
  1. Skip if the path does not exist (warn).
  2. Run `_verify_checksum` against `models/efficientat_mn10_v6_onnx.sha256`. Mismatch → refuse load (T-21-05). Missing manifest → warn and continue (dev workflow convenience).
  3. Build `ort.SessionOptions` with `intra_op_num_threads=num_threads`, `inter_op_num_threads=1`, and the configured execution provider (`CPUExecutionProvider` by default).
  4. Build `ort.InferenceSession`. On failure, continue to next candidate.
- If no candidate loads, raises `RuntimeError` with both paths and the last exception.
- `.classify(mel)` validates `mel.shape == (128, T)`, reshapes to `(1, 1, 128, T)`, runs one session, and returns the logits for batch 0 as a `float32` 1-D vector. For the binary head (`num_classes=1`) this is a `(1,)` array; callers apply `sigmoid` to get the drone probability (per `MODEL_HEAD_NOTES.md`).
- `.active_model_path` and `.num_classes` introspection for Plan 21-07's `/status` endpoint.

Checksum helpers (`_load_expected_checksums`, `_sha256_of`, `_verify_checksum`) handle standard `sha256sum`-format manifests including the `*` binary-mode prefix.

**`apps/rpi-edge/tests/test_inference_latency_host.py`** — 4 GREEN tests (replacing the single-stub Wave 0 entry):

| Test | Check |
|------|-------|
| `test_int8_onnx_inference_under_150ms` | 20-sample p50 latency on a 1 s zero-silence mel input must be < 150 ms on host (Pi 4 budget proxy per 21-VALIDATION.md) |
| `test_fallback_to_fp32_on_missing_int8` | Pointing `onnx_path` at a nonexistent file with valid `fallback_onnx_path` lands on FP32 |
| `test_raises_runtime_error_when_no_model_loads` | Both paths missing → `RuntimeError("Could not load any ONNX model")` |
| `test_classify_output_is_num_classes_vector` | `classify()` returns a `(1,)` `float32` vector (binary head contract from 21-01) |

Result: `pytest tests/test_inference_latency_host.py` → **4 passed**.

## Verification

```
$ cd apps/rpi-edge && python3 -m pytest tests/test_resample_48_to_32.py tests/test_hysteresis.py tests/test_inference_latency_host.py -q
...........                                                              [100%]
11 passed

$ grep -n "from src\|import src\|import torch\|from torch" \
    apps/rpi-edge/skyfort_edge/audio.py \
    apps/rpi-edge/skyfort_edge/hysteresis.py \
    apps/rpi-edge/skyfort_edge/inference.py
(no matches)
```

All success criteria from the plan satisfied:
- [x] audio.py provides 48k→32k resampler and an AudioCapture over sounddevice
- [x] inference.py wraps ORT InferenceSession around the v6 int8 ONNX with binary-head contract
- [x] hysteresis.py implements the K-of-N state machine per D-09 / D-12 / D-14
- [x] `test_resample_48_to_32`, `test_hysteresis`, `test_inference_latency_host` all GREEN
- [x] Each task committed individually (`66ccb76`, `9d095e8`)
- [x] No torch imports in `apps/rpi-edge/skyfort_edge/*.py`
- [x] No `src.acoustic` imports in audio.py / hysteresis.py / inference.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Robustness] Extra constructor guards in `HysteresisStateMachine`**
- The plan's `__init__` only rejected inverted thresholds. I also reject `confirm_hits < 1`, `release_hits < 1`, and `min_on_seconds < 0`. These are correctness requirements: zero or negative hit counts would put the state machine in an impossible regime where either latching or releasing happens instantly / never, and Plan 21-07's config loader cannot meaningfully guard against runtime float drift.
- Files: `apps/rpi-edge/skyfort_edge/hysteresis.py`
- Commit: `66ccb76`

**2. [Rule 2 — Robustness] `test_classify_output_is_num_classes_vector` added beyond the plan**
- The plan's `<behavior>` block for Task 2 says `.classify(mel)` returns a `(num_classes,)` vector. This contract is critical for Plan 21-06 / 21-07 to correctly consume logits into the hysteresis state machine, and it is easy to silently break (e.g. returning `logits[0]` as a scalar for a single-logit head). Added an explicit shape + dtype assertion.
- Files: `apps/rpi-edge/tests/test_inference_latency_host.py`
- Commit: `9d095e8`

**3. [Rule 2 — Robustness] `test_raises_runtime_error_when_no_model_loads` added beyond the plan**
- The plan's `<behavior>` block explicitly says "If both fail → raise RuntimeError." Added a test that exercises this path so future refactors cannot silently swallow the error.
- Files: `apps/rpi-edge/tests/test_inference_latency_host.py`
- Commit: `9d095e8`

**4. [Rule 2 — Robustness] Extra 1 kHz sinusoid round-trip in `test_resample_poly_2_3_correctness`**
- The plan only specified length + dtype + silence preservation. Adding the FFT peak-bin check gives a real-world correctness signal that would catch a sign error or `up`/`down` swap the length check alone would not.
- Files: `apps/rpi-edge/tests/test_resample_48_to_32.py`
- Commit: `66ccb76`

**5. [Rule 1 — Bug guard] Integer-prefix handling in `_load_expected_checksums`**
- Standard `sha256sum` output uses `*` before the filename for binary mode. The plan's parser used `split(maxsplit=1)` and `parts[1].strip()`, which would leave `*efficientat_mn10_v6_int8.onnx` as the recorded name and then fail to find a match for `Path(...).name == "efficientat_mn10_v6_int8.onnx"`. I strip the leading `*` during parse. The committed manifest happens to use the text-mode form (no `*`) so this is defensive, not fixing a current bug.
- Files: `apps/rpi-edge/skyfort_edge/inference.py`
- Commit: `9d095e8`

No Rule 4 (architectural) escalations. No auth gates.

## Authentication Gates

None.

## Threat Flags

No new security-relevant surface introduced beyond the plan's `<threat_model>`:

- **T-21-05 (model tampering):** mitigated by `_verify_checksum` which refuses to load any candidate whose SHA256 disagrees with the committed manifest.
- **T-21-13 (audio callback blocks Python GIL):** mitigated by `AudioCapture._callback` doing only `np.copyto` + integer increments under a lock, with no logging, allocation, or string formatting on the PortAudio thread.

## Known Stubs

None in this plan's outputs. All 11 tests are fully implemented and pass.

Remaining Wave 0 RED stubs **not** owned by 21-05 (owned by Plans 21-06 / 21-07 per `<scope_constraint>`):
- `tests/test_gpio_sigterm.py`
- `tests/test_audio_alarm_degrades.py`
- `tests/test_detection_log.py`
- `tests/test_http_endpoints.py`
- `tests/test_e2e_golden_audio.py`

## Self-Check: PASSED

Files:
- FOUND: `apps/rpi-edge/skyfort_edge/audio.py`
- FOUND: `apps/rpi-edge/skyfort_edge/hysteresis.py`
- FOUND: `apps/rpi-edge/skyfort_edge/inference.py`
- FOUND: `apps/rpi-edge/tests/test_resample_48_to_32.py` (updated, no more `pytest.fail`)
- FOUND: `apps/rpi-edge/tests/test_hysteresis.py` (updated, no more `pytest.fail`)
- FOUND: `apps/rpi-edge/tests/test_inference_latency_host.py` (updated, no more `pytest.fail`)

Commits (verified with `git log --oneline`):
- FOUND: `66ccb76` feat(21-05): AudioCapture + resample 48->32k + hysteresis state machine
- FOUND: `9d095e8` feat(21-05): OnnxClassifier with int8-preferred + FP32 fallback + checksum verify

Test run: `pytest tests/test_resample_48_to_32.py tests/test_hysteresis.py tests/test_inference_latency_host.py -q` → **11 passed**.
