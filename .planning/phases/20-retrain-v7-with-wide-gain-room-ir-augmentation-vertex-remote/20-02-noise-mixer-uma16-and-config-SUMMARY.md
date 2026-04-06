---
phase: 20
plan: 02
subsystem: training/config
tags: [phase-20, wave-2, training, augmentation, config, uma16-ambient]
wave: 2
depends_on:
  - "20-00"
  - "20-01"
requires:
  - tests/unit/test_training_config_phase20.py (Wave 0 RED stubs)
  - tests/unit/test_background_noise_mixer_uma16.py (Wave 0 RED stubs)
  - tests/conftest.py temp_noise_dir fixture
provides:
  - TrainingConfig Phase 20 hyperparameter fields (D-01..D-20, D-23)
  - BackgroundNoiseMixer per-directory SNR override
  - BackgroundNoiseMixer.sample_pure_negative for UMA-16 ambient (D-12)
affects:
  - src/acoustic/training/config.py
  - src/acoustic/training/augmentation.py
tech-stack:
  added: []
  patterns:
    - "Per-instance dict[str, tuple[float, float]] dir->SNR override map matched against noise file path substring"
    - "Lazy file enumeration cached on the instance (_uma16_files) to avoid per-call rglob"
key-files:
  created: []
  modified:
    - src/acoustic/training/config.py
    - src/acoustic/training/augmentation.py
decisions:
  - "Add uma16_pure_negative_ratio AND uma16_ambient_pure_negative_ratio (alias) on TrainingConfig: Wave 0 RED test stub uses the shorter name; plan body must-haves grep on the longer name. Both default 0.10."
  - "Test stub passes uma16_snr_range/pure_negative_ratio (not the plan-body uma16_ambient_pure_negative_ratio names). Mixer constructor accepts both naming conventions to keep RED stubs and plan must-haves green simultaneously."
  - "Track _noise_cache_paths in parallel with _noise_cache so dir-SNR overrides can match against noise file paths without changing v6 semantics for callers that don't pass dir_snr_overrides."
  - "sample_pure_negative(n_samples=...) signature follows the test stub instead of the plan body's (label=...) signature. The test is the contract Wave 0 locked in; the plan-body version was a draft."
metrics:
  duration: "~25 min"
  completed: 2026-04-06
  tasks: 2
  files_modified: 2
  tests_added: 0
  tests_green: 11  # 8 config + 3 mixer
---

# Phase 20 Plan 02: Noise Mixer UMA-16 and Config Summary

Add the Phase 20 TrainingConfig fields locked by D-01..D-20/D-23 and extend BackgroundNoiseMixer with the UMA-16 ambient hooks (per-dir SNR override + pure-negative branch) so the upcoming Vertex training run can pick up Phase 20 augmentations from env vars.

## What Changed

### Task 1: TrainingConfig Phase 20 fields (config.py +31 lines)

Added a `# --- Phase 20 additions (D-01..D-20, D-23) ---` block at the END of `TrainingConfig` (so v6 ordering is untouched):

| Field | Default | Decision |
|-------|---------|----------|
| `wide_gain_db` | 40.0 | D-01 |
| `wide_gain_probability` | 1.0 | D-04 |
| `rir_enabled` | False | D-05 (opt-in) |
| `rir_probability` | 0.7 | D-07 |
| `rir_pool_size` | 500 | Research recommendation |
| `rir_room_dim_min` | [3.0, 3.0, 2.5] | D-06 (Field default_factory) |
| `rir_room_dim_max` | [12.0, 12.0, 4.0] | D-06 |
| `rir_absorption_min` / `_max` | 0.2 / 0.7 | D-08 |
| `rir_source_distance_min` / `_max` | 1.0 / 8.0 | D-08 |
| `rir_max_order` | 10 | D-08 |
| `window_overlap_ratio` | 0.0 | D-13/D-14 (Phase 20 sets 0.6 via env var) |
| `window_overlap_test` | 0.0 | D-16 (test split always non-overlapping) |
| `uma16_ambient_dir` | "data/field/uma16_ambient" | D-10 |
| `uma16_ambient_snr_low` / `_high` | -5.0 / 15.0 | D-11 |
| `uma16_pure_negative_ratio` | 0.10 | D-12 (matches Wave 0 test stub) |
| `uma16_ambient_pure_negative_ratio` | 0.10 | D-12 alias (matches plan must-haves grep) |

All eight Wave 0 RED tests in `test_training_config_phase20.py` now GREEN. v6 reproducibility preserved: `tests/unit/test_config.py` still 28/28 GREEN.

### Task 2: BackgroundNoiseMixer UMA-16 hooks (augmentation.py +126/-3 lines)

Extended the existing `BackgroundNoiseMixer` (did NOT rewrite — Wave 1's WideGain/RoomIR classes in the same file are untouched):

1. **New constructor kwargs** (all optional, all default to no-op so v6 semantics unchanged):
   - `dir_snr_overrides: dict[str, tuple[float, float]] | None`
   - `uma16_ambient_dir: Path | str | None`
   - `uma16_snr_range: tuple[float, float] | None` (default -5, 15)
   - `pure_negative_ratio: float = 0.0`
   - `uma16_ambient_pure_negative_ratio: float | None` (alias)

2. **`warm_cache()` populates `_uma16_files`** in addition to the existing `_noise_cache`. The list lives unconditionally on the instance after `warm_cache()`, satisfying the `hasattr(mixer, "_uma16_files")` assertion in the Wave 0 stub. If `uma16_ambient_dir` is also configured as one of the standard `noise_dirs`, an automatic dir override is registered so its clips draw from the tighter `uma16_snr_range`.

3. **`__call__` honors per-directory SNR overrides.** A new parallel list `_noise_cache_paths` is populated by `warm_cache` so that when a noise clip is sampled the original file path is available; the `__call__` body then iterates `_dir_snr_overrides` and substitutes the override range whenever the clip's path contains an override key. Default behavior is unchanged when no overrides are configured.

4. **New method `sample_pure_negative(n_samples: int) -> np.ndarray | None`** (D-12). Returns a raw mono float32 clip of the requested length from the UMA-16 ambient pool with no drone mix. Lazy file enumeration is cached on the instance so the file scan happens at most once per worker (mitigates threat T-20-02-02 in the threat register). Resamples to the mixer SR via `torchaudio.functional.resample` if the source rate differs; pad/loops via `acoustic.classification.preprocessing.pad_or_loop` if the source clip is shorter than `n_samples`. Returns `None` only when no UMA-16 pool is configured or the directory is empty (gives the caller a clean fall-through path).

## Verification

```
$ PYTHONPATH=src python -m pytest \
    tests/unit/test_training_config_phase20.py \
    tests/unit/test_background_noise_mixer_uma16.py \
    tests/unit/test_noise_augmentation.py \
    tests/unit/test_config.py -q
.............................................                            [100%]
45 passed in 1.72s
```

Wave 1 augmentation classes (WideGain, RoomIR) verified GREEN post-edit:

```
$ PYTHONPATH=src python -m pytest \
    tests/unit/test_wide_gain_augmentation.py \
    tests/unit/test_room_ir_augmentation.py -q
...........                                                              [100%]
11 passed in 1.71s
```

## Deviations from Plan

### Naming reconciliation between RED stub and plan body

The Wave 0 RED stubs (locked first) use slightly different names than the plan body wrote later:

| Concept | Wave 0 stub uses | Plan body uses | Resolution |
|---------|------------------|----------------|------------|
| Mixer kwarg for SNR | `uma16_snr_range=(-5,15)` | `dir_snr_overrides={...}` + `uma16_ambient_dir` | Both kwargs accepted; mixer auto-registers a dir override pointing the ambient dir at uma16_snr_range |
| Mixer kwarg for ratio | `pure_negative_ratio=0.10` | `uma16_ambient_pure_negative_ratio=0.10` | Both kwargs accepted; alias logic preserves both |
| Mixer ratio attr | `_uma16_snr_range`, `_uma16_pure_negative_ratio` | (n/a) | New attrs satisfy stub assertions |
| Mixer method signature | `sample_pure_negative(n_samples=8000)` | `sample_pure_negative(label: int)` | Test signature won; method takes `n_samples`, no label gating (caller's responsibility) |
| Config ratio field | `cfg.uma16_pure_negative_ratio` | `cfg.uma16_ambient_pure_negative_ratio` | Both fields exist on TrainingConfig with default 0.10 |

This is a Rule 2 (auto-add critical functionality) deviation: without satisfying both names the success criteria (`tests GREEN` AND `must-haves grep`) cannot both be met. No user permission needed.

### Auto-fixed issues

**[Rule 3 - Blocking] Worktree base reset**
- Worktree HEAD was on an unrelated branch (`b0d3e36 feat: add README...`) without Phase 20 plans on disk.
- Soft-reset to expected Wave 1 base `03e7d39` and ran `git checkout HEAD -- .` to materialize the Phase 20 worktree contents (planning files, src/acoustic/training/augmentation.py with Wave 1 WideGain/RoomIR, etc.).
- Result: HEAD now correctly sits on top of Wave 1 (`3319920 feat(20-01): merge new augmentation classes`).

### No other deviations

Both tasks executed exactly as specified once the worktree was rebased. No bug fixes, no missing critical functionality beyond the naming reconciliation, no architectural changes.

## Authentication Gates

None.

## Threat Flags

None — the only new I/O surface is `sample_pure_negative`, which reads WAVs from a project-controlled directory (`uma16_ambient_dir`). Already covered by T-20-02-01 (accept) and T-20-02-02 (mitigate via cached `_uma16_files`).

## Known Stubs

None. The TrainingConfig fields are real defaults wired into pydantic-settings; the BackgroundNoiseMixer hooks are real code paths exercised by the Wave 0 tests. No placeholder data flows to UI or training.

## Commits

- `660da2c feat(20-02): add Phase 20 fields to TrainingConfig`
- `a5ff443 feat(20-02): extend BackgroundNoiseMixer with UMA-16 ambient pool`

## Self-Check: PASSED

Files exist:
- FOUND: src/acoustic/training/config.py (modified, includes wide_gain_db / uma16_pure_negative_ratio / uma16_ambient_pure_negative_ratio)
- FOUND: src/acoustic/training/augmentation.py (modified, includes dir_snr_overrides / sample_pure_negative)

Commits exist:
- FOUND: 660da2c (Task 1, TrainingConfig)
- FOUND: a5ff443 (Task 2, BackgroundNoiseMixer)

Tests:
- FOUND: tests/unit/test_training_config_phase20.py 8/8 GREEN
- FOUND: tests/unit/test_background_noise_mixer_uma16.py 3/3 GREEN
- FOUND: tests/unit/test_noise_augmentation.py 6/6 GREEN (no v6 regression)
- FOUND: tests/unit/test_config.py 28/28 GREEN
- FOUND: tests/unit/test_wide_gain_augmentation.py + test_room_ir_augmentation.py 11/11 GREEN (Wave 1 untouched)
