---
phase: 21
plan: 03
subsystem: rpi-edge
tags: [onnx, quantization, model-conversion, sanity-gate]
requires:
  - models/efficientat_mn10_v6.pt (Phase 20 training output)
  - src/acoustic/classification/efficientat/{model,__init__}.py
provides:
  - scripts/convert_efficientat_to_onnx.py
  - models/efficientat_mn10_v6_fp32.onnx
  - models/efficientat_mn10_v6_int8.onnx
  - models/efficientat_mn10_v6_onnx.sha256
affects:
  - apps/rpi-edge/tests/test_onnx_conversion_sanity.py (RED -> GREEN)
  - .gitignore (negation entries for committed v6 .onnx artifacts)
tech-stack:
  added:
    - onnxruntime.quantization (dynamic int8 quant)
    - torch.onnx.export (TorchScript exporter, opset 17)
  patterns:
    - LogitsOnly adapter strips MN's (logits, features) tuple and forces
      consistent (batch, num_classes) shape across all batch sizes.
    - Sanity gate deletes failing artifacts before exit so a broken export
      can never silently land in models/.
key-files:
  created:
    - scripts/convert_efficientat_to_onnx.py
    - models/efficientat_mn10_v6_fp32.onnx
    - models/efficientat_mn10_v6_int8.onnx
    - models/efficientat_mn10_v6_onnx.sha256
  modified:
    - apps/rpi-edge/tests/test_onnx_conversion_sanity.py
    - .gitignore
decisions:
  - D-05: Pi runs ONNX (not torch) — both FP32 and int8 artifacts now exist.
  - D-06: ONNX export uses opset 17, dynamic batch axis, fixed time axis (100 frames).
  - D-07: int8 dynamic quant produced as the default; FP32 retained as fallback.
  - D-08: Pre-write sanity gate deletes broken artifacts and exits non-zero.
metrics:
  duration_minutes: ~30
  completed_date: 2026-04-08
  tasks_completed: 2
  tests_added: 4
  files_created: 4
  files_modified: 2
---

# Phase 21 Plan 03: Convert EfficientAT MN10 v6 to ONNX (FP32 + int8) Summary

Host-side conversion pipeline that turns the trained `efficientat_mn10_v6.pt` checkpoint into the two ONNX artifacts the Pi will ship — gated by a re-runnable sanity check that refuses to write any artifact whose top-1 agreement against the torch reference falls below tolerance.

## What was built

1. **`scripts/convert_efficientat_to_onnx.py`** — single-file conversion CLI:
   - Loads `efficientat_mn10_v6.pt` via `torch.load(weights_only=True)`, constructs MN with the exact same args used in `src/acoustic/classification/efficientat/__init__.py` (`num_classes=1, width_mult=1.0, head_type="mlp", input_dim_f=128, input_dim_t=100`), loads state_dict, sets `eval()`.
   - Wraps the model in a `LogitsOnly` adapter that strips the `(logits, features)` tuple AND reshapes to `(batch, num_classes)` to enforce a consistent rank-2 output shape across all batch sizes (see Bug #1 below).
   - Exports FP32 ONNX with `opset_version=17`, `dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}}`, validates with `onnx.checker.check_model`.
   - Runs the FP32 sanity gate against a deterministic 20-sample mel-shaped batch (`torch.randn` * 0.5, seed=42). Fails closed: deletes the .onnx file and `SystemExit(1)` if top-1 agreement < `--tolerance-fp32` (default 0.99) or mean |logit delta| ≥ `--logit-delta-max` (default 0.05).
   - Quantizes to int8 via `quantize_dynamic(weight_type=QInt8, op_types_to_quantize=["MatMul", "Gemm"])` then re-runs the sanity gate at the relaxed int8 tolerance (default 0.95).
   - Writes a `sha256sum -c`-compatible manifest at `models/efficientat_mn10_v6_onnx.sha256`.
   - Prints a summary table with size, top-1 agreement, mean delta, and sha256 prefix.
   - CLI flags: `--checkpoint`, `--output-dir`, `--skip-int8`, `--validate` (re-validate without re-export), `--tolerance-fp32`, `--tolerance-int8`, `--logit-delta-max`.
   - Docstring contains the verbatim "MatMul and Gemm" caveat note required by D-08.

2. **`models/efficientat_mn10_v6_fp32.onnx`** — 16,850,925 bytes, opset 17, input `mel` shape `[batch, 1, 128, 100]`, output `logits` shape `[batch, 1]`.

3. **`models/efficientat_mn10_v6_int8.onnx`** — 8,704,653 bytes (~48% smaller than FP32). Same I/O shape.

4. **`models/efficientat_mn10_v6_onnx.sha256`** — manifest covering both artifacts; verified with `cd models && sha256sum -c efficientat_mn10_v6_onnx.sha256` → both OK.

5. **`apps/rpi-edge/tests/test_onnx_conversion_sanity.py`** — replaced the RED stub with 4 GREEN tests:
   - `test_fp32_onnx_top1_agreement_ge_99pct` — 50 random mel inputs (seed=1337), top-1 agreement ≥ 99%.
   - `test_int8_onnx_top1_agreement_ge_95pct` — same batch, int8 top-1 agreement ≥ 95% (relaxed per Conv-only quant caveat).
   - `test_mean_logit_delta_below_threshold` — `mean(|fp32_onnx − torch|) < 0.05`.
   - `test_checksum_file_valid` — sha256 manifest matches on-disk bytes for both .onnx files (T-21-05).

## Verification results

| Artifact | Top-1 vs torch | Mean |Δ logit| | Size | sha256 prefix |
|----------|----------------|----------------|------|----------------|
| efficientat_mn10_v6_fp32.onnx | 1.0000 (20/20 in script, 50/50 in test) | 5e-6 | 16.85 MB | (recorded in manifest) |
| efficientat_mn10_v6_int8.onnx | 1.0000 (20/20 in script, 50/50 in test) | 0.0441 | 8.70 MB  | (recorded in manifest) |

- `python scripts/convert_efficientat_to_onnx.py` → exit 0
- `cd models && sha256sum -c efficientat_mn10_v6_onnx.sha256` → both OK
- `python -c "import onnx; onnx.checker.check_model('models/efficientat_mn10_v6_fp32.onnx')"` → OK
- `pytest apps/rpi-edge/tests/test_onnx_conversion_sanity.py` → 4 passed
- Combined GREEN suite (21-02 + 21-03): `test_onnx_conversion_sanity.py + test_preprocess_parity.py + test_preprocess_drift.py` → 6 passed, 3 skipped (preprocess parity has a 3-fixture parametrization where 2 fixtures don't yet exist, unrelated).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] LogitsOnly inconsistent output shape across batch sizes**
- **Found during:** Task 1 first export run.
- **Issue:** `MN._forward_impl` calls `.squeeze()` on the classifier output, which collapses *all* singleton dims. For the binary head (`num_classes=1`) traced at batch=1, the exported ONNX produced an output of rank 0 (`[]`), then at runtime onnxruntime emitted `Expected shape from model of {} does not match actual shape of {20} for output logits` for any batch > 1. The dynamic batch axis was technically working but the declared output shape was wrong, and `bs=1` returned a 0-d numpy array — both unsafe for the Pi inference path (Plan 21-05).
- **Fix:** `LogitsOnly.forward` now ends with `out.reshape(mel.shape[0], self.num_classes)`, forcing a consistent `(batch, num_classes)` tensor regardless of upstream squeezes. Re-export confirmed `output: ['batch', 1]` and `bs=1 -> (1,1)`, `bs=20 -> (20,1)`.
- **Files modified:** `scripts/convert_efficientat_to_onnx.py`
- **Commit:** `aff401a`

**2. [Rule 3 - Blocker] `.gitignore` blocks committed .onnx artifacts**
- **Found during:** Task 1 commit prep.
- **Issue:** `.gitignore` line 36 globally ignores `*.onnx`, so `git add models/efficientat_mn10_v6_*.onnx` was a no-op. The plan REQUIRES committing both files under `models/` so the Pi (and CI) can trust the sha256 manifest.
- **Fix:** Added two targeted negation entries (`!models/efficientat_mn10_v6_fp32.onnx`, `!models/efficientat_mn10_v6_int8.onnx`) plus an explanatory comment. The wildcard `*.onnx` and `*.pt` ignores remain intact for all other model artifacts.
- **Files modified:** `.gitignore`
- **Commit:** `aff401a`

### Manual

None.

### Architectural changes

None. No Rule 4 escalations.

## Authentication Gates

None encountered.

## Decisions made / re-affirmed

- **Sanity-gate tolerances:** kept as plan defaults (`fp32_tol=0.99`, `int8_tol=0.95`, `logit_delta_max=0.05`). Empirically the artifacts come in well under headroom (FP32 delta ~5e-6, int8 delta ~0.044), so future drift up to ~5e-2 in mean logit delta or up to 1% in top-1 agreement will still trip the gate.
- **Binary head top-1 definition:** for `num_classes=1`, `top1 = sign(logit)` (i.e. >0 → drone, ≤0 → not-drone). Both the conversion script's `_top1` helper and the test's `_top1` helper apply this; multi-class fallback uses argmax.
- **Time axis is FIXED at 100 frames** in the exported ONNX (only batch is dynamic). This matches the 1s window at 32 kHz / hop=320 used by both the training code and the Plan 21-02 vendored numpy preprocess.

## Threat Flags

No new threat surface introduced beyond the plan's `<threat_model>`. Mitigations T-21-05 (sha256 tamper detection) and T-21-11 (broken int8 silently lands) are now both *enforced* — at conversion time inside the script, and at CI test time inside `test_onnx_conversion_sanity.py`.

## Known Stubs

None for plan 21-03 itself.

The other rpi-edge test files (`test_hysteresis.py`, `test_http_endpoints.py`, `test_detection_log.py`, `test_resample_48_to_32.py`, `test_inference_latency_host.py`, `test_config_merge.py`) remain Wave 0 RED stubs. Those are owned by Plans 21-04 / 21-05+ per the scope_constraint and were not touched.

## Self-Check: PASSED

- `scripts/convert_efficientat_to_onnx.py` → FOUND
- `models/efficientat_mn10_v6_fp32.onnx` → FOUND (tracked, 16.85 MB)
- `models/efficientat_mn10_v6_int8.onnx` → FOUND (tracked, 8.70 MB)
- `models/efficientat_mn10_v6_onnx.sha256` → FOUND, `sha256sum -c` OK
- `apps/rpi-edge/tests/test_onnx_conversion_sanity.py` → 4 tests GREEN (no remaining `pytest.fail`)
- Commit `aff401a` (Task 1) → FOUND in git log
- Commit `6b02fa9` (Task 2) → FOUND in git log
