# EfficientAT Model Provenance

> Phase 22 Wave 0: locks the current-state truth about which checkpoint
> lives where before v8 promotion. Protects against silently overwriting
> the wrong baseline.

Generated: 2026-04-08

## Checkpoints on Disk (pre-v8)

| File | Size (bytes) | mtime | sha256 | Identity |
|---|---|---|---|---|
| models/efficientat_mn10_v6.pt | 17019638 | 2026-04-06 10:48 | c8828b5d452c19c11f78a7cd5cb5caabc87339aa6c12656f3be9920587be21eb | v6 (Phase 20) |
| models/efficientat_mn10.pt | 17020041 | 2026-04-05 10:34 | 1b9a5162f0e8f0c93dc96ad358003a05ff4fc7407e599e0d65a32866e9bf7b5a | Pre-v6 checkpoint (does NOT match v6 -- different size and hash) |
| models/efficientat_mn10_v8.pt | 16986726 | 2026-04-13 | 02839a1d102fe7ca3116739160d7d9c97e9a025d73dbe7d6cb9afd147a877071 | v8 (Phase 22) — fine-tuned from v6, 1.0s window contract, val_acc=0.994 |
| models/efficientat_mn10_v7.pt | 17019638 | 2026-04-08 07:56 | 421ea22c403470c0e0e8cc79a0b9135e880e439a686b4aea36e30160129eb807 | v7 (regressed -- see .planning/debug/efficientat-v7-regression-vs-v6.md) |
| models/efficientat_mn10_v5.pt | 17019638 | 2026-04-06 04:08 | (same size as v6, separate checkpoint) | v5 (pre-v6) |
| models/efficientat_mn10_v3.pt | 17019638 | 2026-04-05 20:09 | (same size as v6, separate checkpoint) | v3 |
| models/efficientat_mn10_v2.pt | 17020041 | 2026-04-05 14:08 | 37f28715eb9e359830e7cf30ad22104860953f6ba2fc0cc28fda2ffa05ecdd1e | v2 |

## Live Service Loads

**File loaded at runtime:** Configured via `ACOUSTIC_CNN_MODEL_PATH` env var (default: `models/uav_melspec_cnn.onnx`)
**Code path:** `src/acoustic/main.py:409` -- `classifier = load_model(settings.cnn_model_type, settings.cnn_model_path)`
**Default from config:** `src/acoustic/config.py:61` -- `cnn_model_path: str = "models/uav_melspec_cnn.onnx"` (research CNN, NOT EfficientAT)
**Model type config:** `src/acoustic/config.py:62` -- `cnn_model_type: str = "research_cnn"`
**Env var override:** `ACOUSTIC_CNN_MODEL_PATH` sets the model file, `ACOUSTIC_CNN_MODEL_TYPE` sets the model type (e.g., `efficientat_mn10`)

**To run EfficientAT v6 as the live model:**
```bash
ACOUSTIC_CNN_MODEL_PATH=models/efficientat_mn10_v6.pt ACOUSTIC_CNN_MODEL_TYPE=efficientat_mn10 uvicorn acoustic.main:app
```

**Note:** `models/efficientat_mn10.pt` is NOT v6 -- it is a pre-v6 checkpoint with a different hash.
Operators who have been using `efficientat_mn10.pt` are running an older model, not v6.

## Phase 22 Promotion Target

After v8 passes the D-27 gate, `scripts/promote_efficientat.py --version v8` MUST:
1. Verify `models/efficientat_mn10_v8.pt` sha256 against the expected hex from training
2. Copy `models/efficientat_mn10_v8.pt` to `models/efficientat_mn10.pt`
3. Leave `models/efficientat_mn10_v6.pt` alone as rollback

**Do NOT touch:** `models/efficientat_mn10_v6_fp32.onnx`, `models/efficientat_mn10_v6_int8.onnx`
(Phase 21 Pi edge app artifacts).

## IMPORTANT: efficientat_mn10.pt Identity Mismatch

`models/efficientat_mn10.pt` (sha256: 1b9a5162...) does NOT match `efficientat_mn10_v6.pt`
(sha256: c8828b5d...). It matches v2 in size (17020041 bytes) but not hash. This is a pre-v6
checkpoint. **RESOLVE BEFORE PROMOTION:** Before v8 promotion overwrites `efficientat_mn10.pt`,
confirm whether any deployment depends on the current contents. If not, safe to overwrite.

## Rollback Procedure

```bash
cp models/efficientat_mn10_v6.pt models/efficientat_mn10.pt
# restart service
```
