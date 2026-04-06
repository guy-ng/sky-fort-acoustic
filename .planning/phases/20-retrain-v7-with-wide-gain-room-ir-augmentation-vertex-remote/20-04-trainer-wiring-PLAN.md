---
phase: 20
plan: 04
type: execute
wave: 3
depends_on:
  - "20-00"
  - "20-01"
  - "20-02"
  - "20-03"
files_modified:
  - src/acoustic/training/efficientat_trainer.py
autonomous: true
requirements:
  - D-02
  - D-04
  - D-07
  - D-08
  - D-10
  - D-12
  - D-15
  - D-16
  - D-23
must_haves:
  truths:
    - "EfficientATTrainingRunner constructs ComposedAugmentation in the order WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer for the train split"
    - "Val and test splits use NO RIR augmentation (D-08)"
    - "Test split uses non-overlapping windows (hop == window_samples) per D-16"
    - "Train and val splits use overlap window_overlap_ratio when > 0"
    - "Trainer uses split_file_indices to derive disjoint file lists, then constructs three WindowedHFDroneDataset instances"
    - "Existing three-stage recipe (10/15/20 epochs at 1e-3/1e-4/1e-5) is unchanged"
  artifacts:
    - path: src/acoustic/training/efficientat_trainer.py
      provides: "Phase 20 augmentation chain wiring + windowed dataset construction"
      contains: "WindowedHFDroneDataset"
  key_links:
    - from: src/acoustic/training/efficientat_trainer.py
      to: src/acoustic/training/augmentation.py
      via: "_build_train_augmentation() composes WideGainAugmentation, RoomIRAugmentation, AudiomentationsAugmentation, BackgroundNoiseMixer in correct order (D-02, D-07)"
      pattern: "WideGainAugmentation"
    - from: src/acoustic/training/efficientat_trainer.py
      to: src/acoustic/training/parquet_dataset.py
      via: "split_file_indices(num_files, ...) → train_files, val_files, test_files"
      pattern: "split_file_indices"
---

<objective>
Wire the new augmentation classes and the WindowedHFDroneDataset into EfficientATTrainingRunner so
that v7 training reads the new TrainingConfig fields and produces the augmentation chain in the
LOCKED order WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer (per D-02 + D-07).
Construct disjoint train/val/test file index lists via split_file_indices, then build three
WindowedHFDroneDataset instances — train and val with overlap, test without (D-16). Disable RIR on
val and test (D-08).

Purpose: Phase 20's augmentations are useless until the trainer actually invokes them. This is the
single integration point where every CONTEXT.md decision converges into a runnable training loop.

Output: Modified efficientat_trainer.py that consumes Phase 20 config fields, tests pass.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@src/acoustic/training/efficientat_trainer.py
@src/acoustic/training/augmentation.py
@src/acoustic/training/hf_dataset.py
@src/acoustic/training/parquet_dataset.py
@src/acoustic/training/config.py
@tests/unit/test_efficientat_training.py

<interfaces>
After plans 01, 02, 03:

src/acoustic/training/augmentation.py exports:
- WideGainAugmentation(wide_gain_db, p)
- RoomIRAugmentation(sample_rate, pool_size, room_dim_min, room_dim_max, absorption_range, source_distance_range, max_order, p, seed)
- BackgroundNoiseMixer(noise_dirs, snr_range, sample_rate, p, dir_snr_overrides=None, uma16_ambient_dir=None, uma16_ambient_pure_negative_ratio=0.0)
- AudiomentationsAugmentation(...)  (existing)
- ComposedAugmentation(augmentations: list)  (existing)

src/acoustic/training/hf_dataset.py exports:
- WindowedHFDroneDataset(hf_dataset, file_indices, window_samples, hop_samples, mel_config, augmentation, assumed_clip_samples, sample_rate)

src/acoustic/training/parquet_dataset.py exports:
- split_file_indices(num_files, seed, train, val) -> (train_files, val_files, test_files)

src/acoustic/training/config.py TrainingConfig fields (Phase 20):
- wide_gain_db, wide_gain_probability
- rir_enabled, rir_probability, rir_pool_size, rir_room_dim_min, rir_room_dim_max,
  rir_absorption_min, rir_absorption_max, rir_source_distance_min, rir_source_distance_max, rir_max_order
- window_overlap_ratio, window_overlap_test
- uma16_ambient_snr_low, uma16_ambient_snr_high, uma16_ambient_pure_negative_ratio, uma16_ambient_dir
- noise_dirs (existing)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire Phase 20 augmentation chain into EfficientATTrainingRunner</name>
  <files>
    src/acoustic/training/efficientat_trainer.py
  </files>
  <read_first>
    src/acoustic/training/efficientat_trainer.py,
    src/acoustic/training/augmentation.py,
    src/acoustic/training/hf_dataset.py,
    src/acoustic/training/parquet_dataset.py,
    src/acoustic/training/config.py,
    tests/unit/test_efficientat_training.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    EfficientATTrainingRunner now:
    1. Has a private method `_build_train_augmentation(self) -> ComposedAugmentation` that composes
       (in this exact order): WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer.
    2. Has `_build_eval_augmentation(self) -> ComposedAugmentation | None` that excludes RIR (D-08)
       and ideally excludes WideGain too (eval should reflect deployment levels naturally — but
       if v6 path required wide gain on eval, document and keep). For Phase 20: eval gets only
       BackgroundNoiseMixer with the same noise_dirs (no RIR, no wide gain, no audiomentations).
    3. Constructs three WindowedHFDroneDataset instances (train/val/test) using split_file_indices.
    4. Train & val use hop_samples = window_samples * (1 - window_overlap_ratio) when overlap > 0;
       test uses hop_samples = window_samples (D-16).
    5. The existing three-stage recipe (Stage 1/2/3 epochs + LRs) is UNTOUCHED.
    6. Existing test_efficientat_training.py tests still pass; the new test_val_no_rir test passes.
  </behavior>
  <action>
    Open src/acoustic/training/efficientat_trainer.py. Locate the dataset/augmentation construction
    section (search for `HFDroneDataset` or the prior augmentation builder).

    Step 1 — Add imports near the top of the file:
    ```python
    from acoustic.training.augmentation import (
        WideGainAugmentation,
        RoomIRAugmentation,
        AudiomentationsAugmentation,
        BackgroundNoiseMixer,
        ComposedAugmentation,
    )
    from acoustic.training.hf_dataset import WindowedHFDroneDataset
    from acoustic.training.parquet_dataset import split_file_indices
    ```

    Step 2 — Add a private method on the runner class:
    ```python
    def _build_train_augmentation(self) -> ComposedAugmentation:
        cfg = self._config
        augs: list = []

        # Stage 1: wide gain (D-01..D-04)
        if cfg.wide_gain_db > 0:
            augs.append(WideGainAugmentation(
                wide_gain_db=cfg.wide_gain_db,
                p=cfg.wide_gain_probability,
            ))

        # Stage 2: room impulse response (D-05..D-08)
        if cfg.rir_enabled:
            augs.append(RoomIRAugmentation(
                sample_rate=16000,
                pool_size=cfg.rir_pool_size,
                room_dim_min=tuple(cfg.rir_room_dim_min),
                room_dim_max=tuple(cfg.rir_room_dim_max),
                absorption_range=(cfg.rir_absorption_min, cfg.rir_absorption_max),
                source_distance_range=(cfg.rir_source_distance_min, cfg.rir_source_distance_max),
                max_order=cfg.rir_max_order,
                p=cfg.rir_probability,
            ))

        # Stage 3: audiomentations (existing pitch/stretch/small-gain ±6 dB per D-04)
        if cfg.use_audiomentations:
            augs.append(AudiomentationsAugmentation(
                pitch_semitones=cfg.pitch_shift_semitones,
                time_stretch_range=(cfg.time_stretch_min, cfg.time_stretch_max),
                gain_db=cfg.waveform_gain_db,
                p=cfg.augmentation_probability,
            ))

        # Stage 4: background noise (ESC-50 + UrbanSound8K + FSD50K + UMA-16 ambient per D-10/D-18/D-20)
        if cfg.noise_augmentation_enabled and cfg.noise_dirs:
            dir_snr_overrides = {
                "uma16_ambient": (cfg.uma16_ambient_snr_low, cfg.uma16_ambient_snr_high),
            }
            mixer = BackgroundNoiseMixer(
                noise_dirs=[Path(d) for d in cfg.noise_dirs],
                snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
                sample_rate=16000,
                p=cfg.noise_probability,
                dir_snr_overrides=dir_snr_overrides,
                uma16_ambient_dir=cfg.uma16_ambient_dir,
                uma16_ambient_pure_negative_ratio=cfg.uma16_ambient_pure_negative_ratio,
            )
            mixer.warm_cache()
            augs.append(mixer)

        return ComposedAugmentation(augs)

    def _build_eval_augmentation(self) -> ComposedAugmentation | None:
        """Eval pipeline excludes RIR (D-08), wide gain, and audiomentations.
        Eval reflects clean-ish inputs plus BG noise to keep metrics comparable
        across runs. The dedicated real-UMA-16 eval set (D-27) provides the
        deployment-distribution check.
        """
        cfg = self._config
        if not (cfg.noise_augmentation_enabled and cfg.noise_dirs):
            return None
        mixer = BackgroundNoiseMixer(
            noise_dirs=[Path(d) for d in cfg.noise_dirs],
            snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
            sample_rate=16000,
            p=cfg.noise_probability,
        )
        mixer.warm_cache()
        return ComposedAugmentation([mixer])
    ```

    Step 3 — In the dataset construction section (replace the legacy HFDroneDataset path BUT
    only when window_overlap_ratio > 0 or rir_enabled — to preserve v6 reproducibility):
    ```python
    use_phase20_path = (
        self._config.window_overlap_ratio > 0
        or self._config.rir_enabled
        or self._config.wide_gain_db != 6.0  # legacy default sentinel
    )

    if use_phase20_path:
        num_files = len(hf_dataset)
        train_files, val_files, test_files = split_file_indices(
            num_files=num_files,
            seed=self._config.seed,
            train=self._config.train_ratio,
            val=self._config.val_ratio,
        )
        window_samples = int(0.5 * 16000)  # 8000 samples = 0.5 s @ 16 kHz
        train_hop = max(1, int(window_samples * (1.0 - self._config.window_overlap_ratio)))
        test_hop = window_samples  # D-16: non-overlapping test split

        train_dataset = WindowedHFDroneDataset(
            hf_dataset, train_files,
            window_samples=window_samples, hop_samples=train_hop,
            mel_config=mel_config, augmentation=self._build_train_augmentation(),
        )
        val_dataset = WindowedHFDroneDataset(
            hf_dataset, val_files,
            window_samples=window_samples, hop_samples=train_hop,
            mel_config=mel_config, augmentation=self._build_eval_augmentation(),
        )
        test_dataset = WindowedHFDroneDataset(
            hf_dataset, test_files,
            window_samples=window_samples, hop_samples=test_hop,
            mel_config=mel_config, augmentation=self._build_eval_augmentation(),
        )
    else:
        # Legacy v6 path -- unchanged
        ...
    ```

    Step 4 — Confirm the Stage 1/2/3 recipe block (epochs + LRs) is NOT touched. Per D-23 these
    are read from config (`stage1_epochs`, `stage2_epochs`, `stage3_epochs`, `stage1_lr`, etc.)
    and Plan 05 sets them via env vars. The runner does not need code changes for the recipe.

    Important: ensure the dataset class's augmentation is applied at 16 kHz BEFORE resampling to
    32 kHz mel. Per Research §"Wiring the Composed Augmentation" — if the trainer's existing
    `_LazyEfficientATDataset` resamples internally, the augmentation hook needs to live before
    the resample call. WindowedHFDroneDataset receives the augmentation in __init__ and calls
    it inside __getitem__ before any mel computation, which already satisfies this.

    Add a test_val_no_rir test stub if not already in test_efficientat_training.py — assert that
    `_build_eval_augmentation` does NOT include RoomIRAugmentation in its augs list.
  </action>
  <verify>
    <automated>pytest tests/unit/test_efficientat_training.py tests/unit/test_efficientat.py tests/unit/training/test_trainer_augmentation_order.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "_build_train_augmentation" src/acoustic/training/efficientat_trainer.py` returns at least one method definition + caller
    - `grep -n "_build_eval_augmentation" src/acoustic/training/efficientat_trainer.py` returns at least one
    - `grep -n "WideGainAugmentation\|RoomIRAugmentation" src/acoustic/training/efficientat_trainer.py` returns matches
    - `grep -n "WindowedHFDroneDataset" src/acoustic/training/efficientat_trainer.py` returns matches
    - `grep -n "split_file_indices" src/acoustic/training/efficientat_trainer.py` returns matches
    - `grep -n "test_hop = window_samples" src/acoustic/training/efficientat_trainer.py` confirms D-16 non-overlap on test split
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0 (existing + new RIR-disable test pass)
    - `pytest tests/unit/test_efficientat.py -x -q` exits 0 (no v6 regression on the model itself)
    - `pytest tests/unit/training/test_trainer_augmentation_order.py -x -q` exits 0 — confirms the chain order WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer is locked AND eval excludes RIR (D-02, D-07, D-08)
  </acceptance_criteria>
  <done>
    Trainer wiring complete; Phase 20 augmentation chain ordered correctly; sliding-window dataset
    constructed via leakage-safe file split; RIR disabled on val/test; eval+training tests GREEN.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| TrainingConfig env vars → runner | Trainer reads pydantic-loaded config; values originate from local CI or Vertex submission. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-04-01 | Tampering | Augmentation chain order | mitigate | Order is enforced by code structure (sequential augs.append calls). Unit test test_val_no_rir + a future order test in Plan 06 verify. |
| T-20-04-02 | Repudiation (silent v6 path) | use_phase20_path branching | mitigate | Three explicit conditions trigger Phase 20 path; logged at runner startup so the operator can confirm which path is active. |
</threat_model>

<verification>
- Runner dataset/augmentation tests pass
- Runner does not regress on the legacy v6 path (test_efficientat_training.py)
- Augmentation chain order is hard-coded (ComposedAugmentation([wide, rir, audio, bg]))
</verification>

<success_criteria>
- `pytest tests/unit/test_efficientat_training.py tests/unit/test_efficientat.py tests/unit/training/test_trainer_augmentation_order.py -x -q` exits 0
- `tests/unit/training/test_trainer_augmentation_order.py` exits 0 (chain order locked)
- All grep acceptance checks pass
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-04-SUMMARY.md`
</output>
