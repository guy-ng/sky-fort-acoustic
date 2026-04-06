---
phase: 20
plan: 02
type: execute
wave: 2
depends_on:
  - "20-00"
  - "20-01"
files_modified:
  - src/acoustic/training/augmentation.py
  - src/acoustic/training/config.py
autonomous: true
requirements:
  - D-09
  - D-10
  - D-11
  - D-12
  - D-17
  - D-18
  - D-20
  - D-23
must_haves:
  truths:
    - "BackgroundNoiseMixer accepts a per-directory SNR override so UMA-16 ambient uses (-5, +15) dB"
    - "BackgroundNoiseMixer can return a pure-negative branch (~10% of label=0 samples) sourced from uma16_ambient"
    - "TrainingConfig exposes wide_gain_db, rir_enabled, rir_probability, rir_pool_size, rir_*, window_overlap_ratio, window_overlap_test, uma16_ambient_snr_low, uma16_ambient_snr_high, uma16_ambient_pure_negative_ratio"
    - "Default noise_dirs auto-population includes esc50, urbansound8k, fsd50k_subset, uma16_ambient when noise_augmentation_enabled=True"
    - "All Wave 0 unit tests for noise mixer UMA16 and config phase 20 fields are GREEN"
  artifacts:
    - path: src/acoustic/training/config.py
      provides: "TrainingConfig phase 20 field additions"
      contains: "wide_gain_db"
    - path: src/acoustic/training/augmentation.py
      provides: "BackgroundNoiseMixer per-dir SNR override + pure-negative branch"
      contains: "uma16_ambient_snr"
  key_links:
    - from: src/acoustic/training/augmentation.py
      to: src/acoustic/training/config.py
      via: "BackgroundNoiseMixer reads uma16_ambient SNR via constructor params populated by trainer from TrainingConfig"
      pattern: "uma16_ambient_snr"
---

<objective>
Add the new TrainingConfig fields locked by Phase 20 (wide gain, RIR knobs, sliding-window
overlap, UMA-16 ambient SNR + pure-negative ratio, expanded noise_dirs auto-population) and
extend BackgroundNoiseMixer to (a) honor a per-directory SNR override so UMA-16 ambient uses the
tighter (-5, +15) dB range from D-11, and (b) emit a "pure negative" branch where ~10% of label=0
samples are returned as raw UMA-16 ambient (D-12).

Purpose: Phase 20 is mostly orchestration on existing primitives. This plan does the config
plumbing (D-23 hyperparameters and D-20 noise_dirs auto-population) and the only behavior change
required of BackgroundNoiseMixer (per-dir SNR override + pure negative). Without these,
training cannot pick up Phase 20 augmentations from env vars.

Output: Extended TrainingConfig with new fields, modified BackgroundNoiseMixer with override hooks,
all corresponding Wave 0 RED tests turn GREEN.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@src/acoustic/training/config.py
@src/acoustic/training/augmentation.py
@tests/unit/test_training_config_phase20.py
@tests/unit/test_background_noise_mixer_uma16.py

<interfaces>
Existing pydantic-settings TrainingConfig pattern (src/acoustic/training/config.py):
- Fields are class attributes with type + default; pydantic-settings reads ACOUSTIC_TRAINING_<UPPER> env vars
- list[str] env vars are JSON-encoded (pydantic-settings v2)

Existing BackgroundNoiseMixer (src/acoustic/training/augmentation.py:120):
- __init__(self, noise_dirs: list[Path], snr_range: tuple[float, float], sample_rate: int, p: float)
- warm_cache() loads/scans noise files
- __call__(audio) mixes noise at random SNR from snr_range
- Already pickle-safe (per Phase 15)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extend TrainingConfig with Phase 20 fields</name>
  <files>
    src/acoustic/training/config.py
  </files>
  <read_first>
    src/acoustic/training/config.py,
    tests/unit/test_training_config_phase20.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
  </read_first>
  <behavior>
    After this task TrainingConfig exposes the following new fields with defaults:
    - wide_gain_db: float = 40.0
    - wide_gain_probability: float = 1.0
    - rir_enabled: bool = False
    - rir_probability: float = 0.7
    - rir_pool_size: int = 500
    - rir_room_dim_min: list[float] = [3.0, 3.0, 2.5]
    - rir_room_dim_max: list[float] = [12.0, 12.0, 4.0]
    - rir_absorption_min: float = 0.2
    - rir_absorption_max: float = 0.7
    - rir_source_distance_min: float = 1.0
    - rir_source_distance_max: float = 8.0
    - rir_max_order: int = 10
    - window_overlap_ratio: float = 0.0  (Phase 20 sets to 0.6 via env var)
    - window_overlap_test: float = 0.0   (test split always non-overlapping per D-16)
    - uma16_ambient_snr_low: float = -5.0
    - uma16_ambient_snr_high: float = 15.0
    - uma16_ambient_pure_negative_ratio: float = 0.10
    - uma16_ambient_dir: str = "data/field/uma16_ambient"

    Env-var loading via pydantic-settings prefix `ACOUSTIC_TRAINING_` works for all new fields.
    All eight Wave 0 RED tests in test_training_config_phase20.py turn GREEN.
  </behavior>
  <action>
    Open src/acoustic/training/config.py and add the new fields to the TrainingConfig pydantic-settings
    class. Place them in a clearly demarcated `# --- Phase 20 additions ---` block at the END of the
    class body so v6/older configs are unaffected.

    Use the EXACT defaults above (these match D-01..D-20 verbatim — do NOT alter values).

    For list fields use `list[float] = Field(default_factory=lambda: [3.0, 3.0, 2.5])` to avoid
    mutable-default issues.

    If TrainingConfig already has a `noise_dirs: list[str]` field, leave it as-is — Phase 20 will
    populate it via env var (Plan 05 vertex_submit). Do NOT add Phase-20-specific auto-population
    logic here; that belongs to vertex_submit / training runner.

    Do NOT remove or rename any existing fields. v6 reproducibility requires backward compatibility.
  </action>
  <verify>
    <automated>pytest tests/unit/test_training_config_phase20.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "wide_gain_db" src/acoustic/training/config.py` returns one match (the field declaration)
    - `grep -n "rir_enabled" src/acoustic/training/config.py` returns one match
    - `grep -n "rir_probability" src/acoustic/training/config.py` returns match with value 0.7
    - `grep -n "rir_max_order" src/acoustic/training/config.py` shows default 10
    - `grep -n "window_overlap_ratio" src/acoustic/training/config.py` shows default 0.0
    - `grep -n "uma16_ambient_snr_low" src/acoustic/training/config.py` shows -5.0
    - `grep -n "uma16_ambient_pure_negative_ratio" src/acoustic/training/config.py` shows 0.10
    - `pytest tests/unit/test_training_config_phase20.py -x -q` exits 0 (all eight tests GREEN)
    - Existing tests still pass: `pytest tests/unit/test_config.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    All Phase 20 fields added with locked defaults; v6 fields untouched; pydantic-settings env var
    loading works; both new and old config tests are GREEN.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Extend BackgroundNoiseMixer with per-dir SNR override + pure-negative branch</name>
  <files>
    src/acoustic/training/augmentation.py
  </files>
  <read_first>
    src/acoustic/training/augmentation.py,
    tests/unit/test_background_noise_mixer_uma16.py,
    tests/unit/test_noise_augmentation.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
  </read_first>
  <behavior>
    BackgroundNoiseMixer gains:
    - Optional constructor parameter `dir_snr_overrides: dict[str, tuple[float, float]] | None = None`
      that maps a noise directory basename or path substring to a custom (snr_low, snr_high) range.
      When sampling a noise clip from a directory matching one of the override keys, use the override
      range instead of the default `snr_range`.
    - Optional constructor parameter `uma16_ambient_dir: str | None = None` and
      `uma16_ambient_pure_negative_ratio: float = 0.0`.
    - New method `sample_pure_negative(label: int) -> np.ndarray | None` that:
        - if label != 0: returns None
        - else with probability `uma16_ambient_pure_negative_ratio` returns a raw mono 16k clip
          loaded from a random file under uma16_ambient_dir (no drone, no SNR mix)
        - otherwise returns None (caller falls through to normal logic)
    - All existing v6 tests still pass (test_noise_augmentation.py).

    All Wave 0 RED tests in test_background_noise_mixer_uma16.py turn GREEN.
  </behavior>
  <action>
    Modify the existing BackgroundNoiseMixer class in src/acoustic/training/augmentation.py.
    Do NOT rewrite the class from scratch — extend it.

    Step 1 — Add new constructor parameters with defaults:
    ```python
    def __init__(
        self,
        noise_dirs: list[Path],
        snr_range: tuple[float, float],
        sample_rate: int = 16000,
        p: float = 0.8,
        dir_snr_overrides: dict[str, tuple[float, float]] | None = None,
        uma16_ambient_dir: str | None = None,
        uma16_ambient_pure_negative_ratio: float = 0.0,
    ):
        ...
        self._dir_snr_overrides = dir_snr_overrides or {}
        self._uma16_ambient_dir = Path(uma16_ambient_dir) if uma16_ambient_dir else None
        self._uma16_pure_negative_ratio = float(uma16_ambient_pure_negative_ratio)
    ```

    Step 2 — In the noise-sampling code path inside `__call__`, AFTER selecting a noise file,
    check whether its parent directory path contains any override key; if yes, use the override
    range to sample SNR instead of `self._snr_range`. Concretely:

    ```python
    snr_lo, snr_hi = self._snr_range
    for key, (lo, hi) in self._dir_snr_overrides.items():
        if key in str(noise_path):
            snr_lo, snr_hi = lo, hi
            break
    snr_db = self._rng.uniform(snr_lo, snr_hi)
    ```

    Step 3 — Add a new public method:
    ```python
    def sample_pure_negative(self, label: int) -> np.ndarray | None:
        """Return a raw UMA-16 ambient clip with probability uma16_pure_negative_ratio
        when label == 0. Otherwise return None.

        Implements D-12: ~10% of negative mini-batch sourced from UMA-16 ambient as
        pure label=0 samples (no drone mix).
        """
        if label != 0 or self._uma16_ambient_dir is None:
            return None
        if self._rng.random() >= self._uma16_pure_negative_ratio:
            return None
        # Lazy file enumeration (cached after first call)
        if not hasattr(self, "_uma16_files") or self._uma16_files is None:
            self._uma16_files = sorted(self._uma16_ambient_dir.rglob("*.wav"))
        if not self._uma16_files:
            return None
        path = self._uma16_files[self._rng.integers(len(self._uma16_files))]
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != self._sr:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, self._sr, sr).astype(np.float32)
        return audio
    ```

    Step 4 — Update __getstate__/__setstate__ if present to include the new fields. If pickling
    is currently handled implicitly (no custom dunder methods), no change needed because the new
    attributes are picklable primitives.
  </action>
  <verify>
    <automated>pytest tests/unit/test_background_noise_mixer_uma16.py tests/unit/test_noise_augmentation.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "dir_snr_overrides" src/acoustic/training/augmentation.py` returns at least one match in BackgroundNoiseMixer
    - `grep -n "uma16_ambient_pure_negative_ratio" src/acoustic/training/augmentation.py` returns matches
    - `grep -n "def sample_pure_negative" src/acoustic/training/augmentation.py` returns one match
    - `pytest tests/unit/test_background_noise_mixer_uma16.py -x -q` exits 0 (all three tests GREEN)
    - `pytest tests/unit/test_noise_augmentation.py -x -q` exits 0 (no v6 regressions)
  </acceptance_criteria>
  <done>
    BackgroundNoiseMixer accepts dir_snr_overrides + uma16 pure-negative params; sample_pure_negative
    method works; existing tests still pass.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Filesystem (uma16_ambient_dir) → BackgroundNoiseMixer | WAV files loaded by soundfile from a project-controlled directory. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-02-01 | Tampering | uma16_ambient_dir WAV files | accept | Files originate from project hardware; no untrusted external input. |
| T-20-02-02 | DoS | sample_pure_negative file scan | mitigate | Cache file list on first call (`self._uma16_files`); no per-call rglob. |
| T-20-02-03 | Information Disclosure | TrainingConfig env vars | accept | Local-only config; no secrets in new Phase 20 fields. |
</threat_model>

<verification>
- TrainingConfig phase 20 tests GREEN
- BackgroundNoiseMixer UMA-16 tests GREEN
- Existing test_noise_augmentation.py and test_config.py still GREEN
</verification>

<success_criteria>
- `pytest tests/unit/test_training_config_phase20.py tests/unit/test_background_noise_mixer_uma16.py tests/unit/test_noise_augmentation.py tests/unit/test_config.py -x -q` exits 0
- All new fields traceable via grep with their exact default values
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-02-SUMMARY.md`
</output>
