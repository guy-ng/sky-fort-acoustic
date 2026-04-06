---
phase: 20
plan: 00
type: execute
wave: 0
depends_on: []
files_modified:
  - tests/unit/test_wide_gain_augmentation.py
  - tests/unit/test_room_ir_augmentation.py
  - tests/unit/test_background_noise_mixer_uma16.py
  - tests/unit/test_sliding_window_dataset.py
  - tests/unit/test_training_config_phase20.py
  - tests/unit/test_vertex_submit_phase20.py
  - tests/unit/test_promotion_gate.py
  - tests/integration/test_vertex_dockerfile_copy.py
  - tests/unit/training/test_trainer_augmentation_order.py
  - tests/conftest.py
  - data/field/uma16_ambient/.gitkeep
  - data/noise/fsd50k_subset/.gitkeep
  - data/eval/uma16_real/.gitkeep
  - data/eval/uma16_real/labels.json.example
autonomous: false
requirements:
  - D-01
  - D-05
  - D-09
  - D-10
  - D-13
  - D-15
  - D-18
  - D-19
  - D-23
  - D-24
  - D-27
  - D-29
must_haves:
  truths:
    - "Every Phase 20 production class has at least one failing test stub before implementation"
    - "UMA-16 ambient WAVs (≥30 min) exist on disk in mono 16 kHz under data/field/uma16_ambient/"
    - "FSD50K subset WAVs exist under data/noise/fsd50k_subset/ for the 6 target classes"
    - "Real-capture eval set ≥20 min with labels.json exists under data/eval/uma16_real/"
  artifacts:
    - path: tests/unit/test_wide_gain_augmentation.py
      provides: "RED stubs for D-01..D-04 WideGainAugmentation behavior"
    - path: tests/unit/test_room_ir_augmentation.py
      provides: "RED stubs for D-05..D-08 RoomIRAugmentation behavior"
    - path: tests/unit/test_sliding_window_dataset.py
      provides: "RED stubs for D-13..D-16 including session-level leakage assertion"
    - path: tests/conftest.py
      provides: "Shared fixtures: synthetic 16kHz waveform, tiny RIR, temp noise dir"
    - path: data/field/uma16_ambient/.gitkeep
      provides: "Directory exists; populated by manual checkpoint with ≥30 min audio"
  key_links:
    - from: tests/unit/test_sliding_window_dataset.py
      to: src/acoustic/training/hf_dataset.py
      via: "import WindowedHFDroneDataset and assert no file_idx leakage across splits"
      pattern: "test_no_file_leakage"
---

<objective>
Wave 0 establishes the test scaffolding and data pre-requisites for Phase 20. Per the Nyquist Rule
in 20-VALIDATION.md, every downstream task must verify against an automated command that exists.
This plan creates RED test stubs for all new classes (WideGainAugmentation, RoomIRAugmentation,
WindowedHFDroneDataset, expanded BackgroundNoiseMixer, TrainingConfig phase 20 fields,
vertex_submit env propagation, promotion gate) and gates the manual data-collection prerequisites
that block training (UMA-16 ambient ≥30 min per D-09, FSD50K subset per D-18, real-capture eval set
≥20 min per D-27).

Purpose: Make every Wave 1+ task verifiable in <60s and ensure Wave 2 (Vertex submit) cannot be
attempted before required training/eval data exists.

Output: 8 RED test files, shared fixtures, 3 data directory checkpoints (manual capture).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-VALIDATION.md
@src/acoustic/training/augmentation.py
@src/acoustic/training/hf_dataset.py
@src/acoustic/training/config.py
@tests/unit/test_augmentation.py
@tests/unit/test_hf_dataset.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Shared fixtures + WideGain/RoomIR/Config test stubs (RED)</name>
  <files>
    tests/conftest.py,
    tests/unit/test_wide_gain_augmentation.py,
    tests/unit/test_room_ir_augmentation.py,
    tests/unit/test_training_config_phase20.py
  </files>
  <read_first>
    tests/unit/test_augmentation.py,
    tests/unit/test_noise_augmentation.py,
    src/acoustic/training/augmentation.py,
    src/acoustic/training/config.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    tests/conftest.py provides fixtures:
      - synthetic_waveform(): np.ndarray float32 mono 16000 Hz, 1.0s, sine 1 kHz amplitude 0.1
      - tiny_rir(): np.ndarray float32, length 800 (50 ms @ 16k), exponential decay
      - temp_noise_dir(tmp_path): writes 5 short WAV files into tmp_path/noise/

    test_wide_gain_augmentation.py (RED — class does not exist yet):
      - test_emits_within_clipping_bounds: instantiate WideGainAugmentation(wide_gain_db=40.0, p=1.0); apply to synthetic_waveform; assert max(abs(out)) <= 1.0
      - test_gain_range_uniform: 1000 calls; assert observed dB spans at least [-30, +30] (subset of ±40)
      - test_probability_zero_pass_through: p=0.0 returns input unchanged (np.array_equal)
      - test_dtype_preserved: input float32, output float32
      - test_pickle_safe: pickle.dumps + loads round-trip works (DataLoader num_workers safety)

    test_room_ir_augmentation.py (RED — class does not exist yet):
      - test_pool_built_at_init: RoomIRAugmentation(pool_size=8, p=1.0) → len(pool) == 8
      - test_output_length_preserved: out.shape == input.shape after __call__
      - test_probability_zero_pass_through: p=0.0 returns input unchanged
      - test_dtype_preserved: float32 in/out
      - test_pickle_safe: pickle round-trip
      - test_max_order_bounded: pool builds in <30s with pool_size=8, max_order=10

    test_training_config_phase20.py (RED — fields do not exist yet):
      - test_wide_gain_db_default: TrainingConfig().wide_gain_db == 40.0
      - test_rir_enabled_default_false: TrainingConfig().rir_enabled is False
      - test_rir_probability_default: TrainingConfig().rir_probability == 0.7
      - test_rir_pool_size_default: == 500
      - test_window_overlap_ratio_default: == 0.0  (opt-in; Phase 20 sets via env var)
      - test_uma16_ambient_snr_range: low == -5.0, high == 15.0
      - test_uma16_pure_negative_ratio: == 0.10
      - test_env_var_override: monkeypatch ACOUSTIC_TRAINING_RIR_ENABLED=true, ACOUSTIC_TRAINING_WIDE_GAIN_DB=20.0 → loaded values match
  </behavior>
  <action>
    Create tests/conftest.py with the three pytest fixtures above. Use scope="function" for tmp_path-derived fixtures and scope="session" for synthetic_waveform/tiny_rir.

    Create the three RED test files. Each test imports from src.acoustic.training.augmentation
    (WideGainAugmentation, RoomIRAugmentation) and src.acoustic.training.config (TrainingConfig).
    These imports MUST currently fail with ImportError or AttributeError — confirming RED state.

    Use exact decision values from CONTEXT.md verbatim:
      - wide_gain_db = 40.0 (D-01)
      - rir_probability = 0.7 (D-07)
      - rir_pool_size = 500 (Discretion + Research recommendation)
      - rir_max_order = 10 (Research Pitfall 3)
      - uma16_ambient_snr = (-5.0, 15.0) (D-11)
      - uma16_pure_negative_ratio = 0.10 (D-12)
  </action>
  <verify>
    <automated>pytest tests/unit/test_wide_gain_augmentation.py tests/unit/test_room_ir_augmentation.py tests/unit/test_training_config_phase20.py --collect-only -q 2>&1 | grep -E "test_" | wc -l</automated>
  </verify>
  <acceptance_criteria>
    - File tests/conftest.py contains `def synthetic_waveform`, `def tiny_rir`, `def temp_noise_dir`
    - File tests/unit/test_wide_gain_augmentation.py contains `test_emits_within_clipping_bounds`, `test_pickle_safe`
    - File tests/unit/test_room_ir_augmentation.py contains `test_pool_built_at_init`, `test_output_length_preserved`
    - File tests/unit/test_training_config_phase20.py contains `test_rir_enabled_default_false`, `test_env_var_override`
    - `pytest tests/unit/test_wide_gain_augmentation.py --collect-only` collects ≥5 tests
    - Running these tests EXITS NON-ZERO (RED state confirmed) — they should fail on ImportError, NOT on assertion mismatch
  </acceptance_criteria>
  <done>
    Three RED test files exist; pytest collects all expected tests; tests fail on import (RED state); conftest.py provides the three shared fixtures.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Sliding-window dataset, BG mixer UMA16, Vertex submit, promotion, Dockerfile test stubs (RED)</name>
  <files>
    tests/unit/test_sliding_window_dataset.py,
    tests/unit/test_background_noise_mixer_uma16.py,
    tests/unit/test_vertex_submit_phase20.py,
    tests/unit/test_promotion_gate.py,
    tests/integration/test_vertex_dockerfile_copy.py,
    tests/unit/training/test_trainer_augmentation_order.py
  </files>
  <read_first>
    src/acoustic/training/hf_dataset.py,
    src/acoustic/training/parquet_dataset.py,
    src/acoustic/training/augmentation.py,
    scripts/vertex_submit.py,
    Dockerfile.vertex,
    tests/unit/test_hf_dataset.py,
    tests/unit/test_noise_augmentation.py,
    tests/unit/test_parquet_dataset.py
  </read_first>
  <behavior>
    test_sliding_window_dataset.py (RED — D-13..D-16):
      - test_window_count_for_uniform_clip: 16000-sample file, window=8000, hop=3200 → 1 + (16000-8000)//3200 = 3 windows
      - test_idx_mapping_consistent: __getitem__(0) returns (file_idx, offset) tuple via internal _items list
      - test_no_file_leakage_across_splits: build dataset with 100 fake files, split file indices 70/15/15, expand to windows; assert intersection of file_idx sets between train/val/test == empty set (CRITICAL — D-15)
      - test_test_split_no_overlap: hop_samples == window_samples on test split (D-16)
      - test_train_val_overlap: hop_samples < window_samples on train and val splits

    test_background_noise_mixer_uma16.py (RED — D-10, D-11):
      - test_uma16_ambient_dir_accepted: BackgroundNoiseMixer accepts data/field/uma16_ambient as one of its noise_dirs
      - test_uma16_specific_snr_range: when source is from uma16_ambient subdir, SNR sampled from (-5.0, 15.0) NOT default (-10, 20)
      - test_pure_negative_branch: pure_negative_ratio=0.10 → ~10% of label=0 samples returned as raw uma16 ambient (no drone mix)

    test_vertex_submit_phase20.py (RED — D-21, D-23, D-24):
      - test_v7_job_name: built job name contains "v7"
      - test_env_vars_include_phase20: ACOUSTIC_TRAINING_WIDE_GAIN_DB, _RIR_ENABLED, _RIR_PROBABILITY, _NOISE_DIRS, _WINDOW_OVERLAP_RATIO present in env_vars dict
      - test_l4_with_t4_fallback: machine_type "g2-standard-8", accelerator NVIDIA_L4 with T4 fallback path
      - test_preflight_quota_check_callable: function `check_l4_quota(project, region) -> bool` exists and returns bool

    test_promotion_gate.py (RED — D-29):
      - test_promotion_blocked_when_dads_fails: dads_acc=0.93, real_tpr=0.85, real_fpr=0.03 → promote() returns False
      - test_promotion_blocked_when_real_tpr_fails: dads_acc=0.97, real_tpr=0.70, real_fpr=0.03 → False
      - test_promotion_blocked_when_real_fpr_fails: dads_acc=0.97, real_tpr=0.85, real_fpr=0.10 → False
      - test_promotion_succeeds_when_both_pass: dads_acc=0.97, real_tpr=0.85, real_fpr=0.03 → True
      - test_promotion_copies_checkpoint: when True, models/efficientat_mn10_v7.pt is copied to models/efficientat_mn10.pt

    test_vertex_dockerfile_copy.py (integration, RED — D-24):
      - test_dockerfile_copies_noise_dir: parse Dockerfile.vertex; assert it contains a `COPY` line for `data/noise` (or its base image inheritance)
      - test_dockerfile_copies_uma16_ambient: assert COPY for `data/field/uma16_ambient`

    test_trainer_augmentation_order.py (RED — D-02, D-07, D-08, locks chain order before Plan 20-04 wires it):
      - test_train_chain_order: instantiate EfficientATTrainingRunner with a phase-20 config; call _build_train_augmentation(); assert
        `[type(a).__name__ for a in train_aug._augmentations] == ["WideGainAugmentation", "RoomIRAugmentation", "AudiomentationsAugmentation", "BackgroundNoiseMixer"]`
      - test_eval_chain_excludes_rir: call _build_eval_augmentation(); assert `"RoomIRAugmentation" not in [type(a).__name__ for a in eval_aug._augmentations]`
      - This test imports `from acoustic.training.efficientat_trainer import EfficientATTrainingRunner` and will RED-fail until Plan 20-04 ships the two private builders.
  </behavior>
  <action>
    Create the five RED test files. All imports MUST currently fail (classes / functions / Dockerfile lines do not yet exist). Use exact thresholds from D-26/D-27/D-29 verbatim:
      - DADS test acc threshold = 0.95
      - Real-capture TPR threshold = 0.80
      - Real-capture FPR threshold = 0.05

    For test_no_file_leakage_across_splits use the pattern from 20-RESEARCH.md "Pitfall 1":
      build sets train_files, val_files, test_files; assert `set(train_files) & set(val_files) == set()`
      AND `set(train_files) & set(test_files) == set()` AND `set(val_files) & set(test_files) == set()`
      AND that for any `idx` in train_dataset._items, `_items[idx][0] in train_files`.

    For test_vertex_dockerfile_copy.py read Dockerfile.vertex as text and use simple substring/regex matches.

    Imports needed (these will fail RED until later plans):
      - from src.acoustic.training.hf_dataset import WindowedHFDroneDataset
      - from src.acoustic.training.parquet_dataset import split_file_indices
      - from scripts.vertex_submit import build_env_vars_v7, check_l4_quota
      - from src.acoustic.evaluation.promotion import promote_v7_if_gates_pass
      - from acoustic.training.efficientat_trainer import EfficientATTrainingRunner   # for augmentation order test

    For test_trainer_augmentation_order.py: create the file under tests/unit/training/ (mkdir + add __init__.py if needed). The
    test must construct a phase-20 TrainingConfig with wide_gain_db=40.0, rir_enabled=True, noise_augmentation_enabled=True, and
    a noise_dirs list pointing at the temp_noise_dir fixture so BackgroundNoiseMixer can instantiate. Patch warm_cache to a no-op
    if needed for the RED stub. The chain-order assertion is the contract — Plan 20-04 implements against it.
  </action>
  <verify>
    <automated>pytest tests/unit/test_sliding_window_dataset.py tests/unit/test_background_noise_mixer_uma16.py tests/unit/test_vertex_submit_phase20.py tests/unit/test_promotion_gate.py tests/integration/test_vertex_dockerfile_copy.py tests/unit/training/test_trainer_augmentation_order.py --collect-only -q 2>&1 | grep -cE "test_"</automated>
  </verify>
  <acceptance_criteria>
    - All six files exist
    - tests/unit/test_sliding_window_dataset.py contains exact string `test_no_file_leakage_across_splits`
    - tests/unit/test_promotion_gate.py contains all four threshold tests with literals 0.95, 0.80, 0.05
    - tests/unit/test_vertex_submit_phase20.py contains `test_l4_with_t4_fallback`
    - tests/unit/training/test_trainer_augmentation_order.py contains both `test_train_chain_order` and `test_eval_chain_excludes_rir`
    - `pytest --collect-only` finds at least 20 tests total across the six files
    - Tests fail on ImportError (RED state)
  </acceptance_criteria>
  <done>
    Six RED test files exist; collection succeeds; tests fail on missing imports. Augmentation chain order is locked into a test before Plan 20-04 begins.
  </done>
</task>

<task type="checkpoint:human-action" gate="blocking">
  <name>Task 3: Manual data acquisition (UMA-16 ambient ≥30 min, real-capture eval ≥20 min, FSD50K subset)</name>
  <what-built>
    Wave 0 test stubs (Tasks 1-2). The three remaining Wave 0 prerequisites require physical
    hardware capture and external data download — Claude cannot automate them.
  </what-built>
  <how-to-verify>
    Three data acquisitions are required. Claude will automate WHAT IT CAN, then this checkpoint verifies the human-required pieces.

    1. UMA-16 AMBIENT (D-09) — ≥30 min mono 16 kHz WAV:
       - Use the existing field-recording UI (Phase 10) or `arecord` + a conversion script.
       - Record FOUR conditions: indoor quiet, indoor with HVAC, outdoor quiet, outdoor with wind.
       - Convert each to mono 16 kHz: `python -c "import soundfile as sf, numpy as np; a, sr = sf.read('in.wav'); m = a.mean(axis=1) if a.ndim>1 else a; from scipy.signal import resample_poly; out = resample_poly(m, 16000, sr).astype(np.float32); sf.write('out.wav', out, 16000)"`
       - Place under `data/field/uma16_ambient/{indoor_quiet,indoor_hvac,outdoor_quiet,outdoor_wind}/*.wav`
       - VERIFY: `python -c "import soundfile as sf, glob; total = sum(sf.info(p).duration for p in glob.glob('data/field/uma16_ambient/**/*.wav', recursive=True)); print(f'{total:.1f}s'); assert total >= 1800, 'need ≥30 min'"`

    2. REAL-CAPTURE EVAL SET (D-27) — ≥20 min UMA-16 with labels.json (≥5 min drone, ≥15 min ambient):
       - Capture drone flight recordings + ambient with the UMA-16 array.
       - Hand-label segments. Create `data/eval/uma16_real/labels.json`:
         ```json
         [
           {"file": "clip_001.wav", "label": "drone", "start_s": 0.0, "end_s": 5.0},
           {"file": "clip_002.wav", "label": "no_drone", "start_s": 0.0, "end_s": 30.0}
         ]
         ```
       - VERIFY: `python -c "import json, soundfile as sf; e = json.load(open('data/eval/uma16_real/labels.json')); drone = sum(x['end_s']-x['start_s'] for x in e if x['label']=='drone'); nd = sum(x['end_s']-x['start_s'] for x in e if x['label']=='no_drone'); print(f'drone={drone:.1f}s nodrone={nd:.1f}s'); assert drone>=300 and nd>=900"`

    3. DRONEAUDIOSET EXPLORATION (D-19) — opportunistic, time-boxed to 30 minutes:
       - Attempt: `python -c "from datasets import load_dataset; ds = load_dataset('ahlab-drone-project/DroneAudioSet', split='train', streaming=True); print(next(iter(ds)).keys())"` to enumerate splits/columns.
       - Look for a clean non-drone subset (column or split). If found, snapshot ≤200 clips into `data/noise/dronaudioset_subset/` for use as additional negatives.
       - If unavailable, license-restricted, or only embedded in mixed recordings → drop silently per Q1 RESOLUTION. Log outcome in 20-00-SUMMARY.md.
       - VERIFY: either `find data/noise/dronaudioset_subset -name '*.wav' | wc -l` ≥ 50, OR user typed `skip-dronaudioset`.

    4. FSD50K SUBSET (D-18) — Wind, Rain, Traffic_noise_and_roadway_noise, Mechanical_fan, Engine, Bird:
       - Option A (preferred): `pip install soundata && python -c "import soundata; ds = soundata.initialize('fsd50k', data_home='data/noise/fsd50k_subset'); ds.download(partial_download=['Wind','Rain','Traffic_noise_and_roadway_noise','Mechanical_fan','Engine','Bird'])"`
       - Option B (manual): download from https://zenodo.org/records/4060432 and filter using `FSD50K.ground_truth/dev.csv`.
       - Place WAVs in `data/noise/fsd50k_subset/{class}/*.wav` (mono 16 kHz preferred but BackgroundNoiseMixer will resample)
       - VERIFY: `find data/noise/fsd50k_subset -name '*.wav' | wc -l` ≥ 200

    BEFORE this checkpoint, Claude should create the empty directories with .gitkeep and a `data/eval/uma16_real/labels.json.example` file.
  </how-to-verify>
  <resume-signal>
    Type "data ready" once all four verification commands above pass, OR "skip-fsd50k" if FSD50K download is deferred, OR "skip-dronaudioset" if DroneAudioSet (D-19) cannot be sourced. Both skip signals are independent and may be combined (e.g. "data ready skip-dronaudioset"). UMA-16 ambient and the real-capture eval set remain HARD blockers for D-27 and v7 promotion and have no skip path.
  </resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Manual data ingest → repo | Audio files come from a hardware capture step; the only "untrusted input" is potentially malformed WAV. |
| Test files → CI | Standard internal threat surface — none unique to this plan. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-00-01 | Tampering | UMA-16 ambient WAV files | accept | Files originate from project-owned hardware; verify via duration assertion only. No external untrusted input. |
| T-20-00-02 | Information Disclosure | data/eval/uma16_real/labels.json | accept | Labels are project-internal recordings; no PII. Stored in repo (gitignored if user prefers — recommend `.gitignore` entry but not blocking). |
| T-20-00-03 | DoS | FSD50K download (~24 GB full set) | mitigate | Use partial_download for 6 classes only (~2-4 GB). Verify free disk space before fetch. |
</threat_model>

<verification>
- All 8 RED test files exist and are collected by pytest
- conftest.py exposes the three required fixtures
- Manual checkpoint passes: ≥30 min UMA-16 ambient, ≥20 min labeled real-capture eval, FSD50K subset present
</verification>

<success_criteria>
- `pytest tests/unit/test_wide_gain_augmentation.py tests/unit/test_room_ir_augmentation.py tests/unit/test_training_config_phase20.py tests/unit/test_sliding_window_dataset.py tests/unit/test_background_noise_mixer_uma16.py tests/unit/test_vertex_submit_phase20.py tests/unit/test_promotion_gate.py tests/integration/test_vertex_dockerfile_copy.py --collect-only -q` succeeds with ≥30 collected tests
- `find data/field/uma16_ambient -name '*.wav' | xargs -I{} python -c "import soundfile as sf; print(sf.info('{}').duration)" | awk '{s+=$1} END {exit !(s>=1800)}'` exits 0
- `test -f data/eval/uma16_real/labels.json` and the duration script above passes
- `find data/noise/fsd50k_subset -name '*.wav' | wc -l` ≥ 200 OR user typed "skip-fsd50k"
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-00-SUMMARY.md`
</output>
