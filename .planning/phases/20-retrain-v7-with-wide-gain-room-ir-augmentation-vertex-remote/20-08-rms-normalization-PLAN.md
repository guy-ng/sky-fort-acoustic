---
phase: 20
plan: 08
type: execute
wave: 3
depends_on:
  - "20-00"
  - "20-04"
  - "20-07"
files_modified:
  - src/acoustic/classification/preprocessing.py
  - src/acoustic/training/augmentation.py
  - src/acoustic/training/efficientat_trainer.py
  - src/acoustic/training/hf_dataset.py
  - src/acoustic/training/config.py
  - src/acoustic/config.py
requirements:
  - D-34
must_haves:
  truths:
    - "_rms_normalize(audio, target=0.1, eps=1e-6) function exists in a shared location, used by both training and inference paths"
    - "RawAudioPreprocessor.process() applies _rms_normalize as the LAST step before return (after resample, after legacy input_gain)"
    - "AcousticSettings.cnn_input_gain default is changed to 1.0 (legacy 500.0 default removed); existing constructor still accepts the value for backwards compat but normalization makes it irrelevant"
    - "WindowedHFDroneDataset.__getitem__ applies _rms_normalize as the LAST step in the augmentation chain — AFTER BackgroundNoiseMixer so SNR-mixed signals are normalized as a unit"
    - "Legacy HFDroneDataset path also gets the same _rms_normalize step (defensive — both code paths must match)"
    - "rms_normalize_target is a config field on TrainingConfig (default 0.1) and a constructor arg on RawAudioPreprocessor (default 0.1); both default to the same anchor"
    - "Eval-time normalization is identical to train-time normalization (no branching). Train/val/test all RMS-normalize."
  artifacts:
    - path: src/acoustic/classification/preprocessing.py
      provides: "_rms_normalize() helper + RawAudioPreprocessor calls it after resample"
      contains: "_rms_normalize"
    - path: src/acoustic/training/augmentation.py
      provides: "RmsNormalize augmentation class — last in the ComposedAugmentation chain"
      contains: "RmsNormalize"
  key_links:
    - from: src/acoustic/classification/preprocessing.py
      to: src/acoustic/training/augmentation.py
      via: "Both call the same _rms_normalize(audio, target, eps) implementation"
      pattern: "_rms_normalize"
    - from: src/acoustic/training/efficientat_trainer.py
      to: src/acoustic/training/augmentation.py
      via: "_build_train_augmentation() appends RmsNormalize as the LAST stage in ComposedAugmentation"
      pattern: "RmsNormalize"
---

<objective>
Implement D-34: per-sample RMS normalization on BOTH the trainer dataset path AND
`RawAudioPreprocessor.process()`, with the same target RMS (0.1) on both sides. This
closes the two coupled bugs surfaced by `scripts/verify_rms_domain_mismatch.py`:

  1. **Train/inference domain shift** — DADS raw RMS ~0.18 vs live UMA-16 post-`cnn_input_gain=500`
     RMS ~9.3 (52x ratio, +1.10 normalized log-mel mean shift). The model never sees the
     inference distribution.
  2. **DADS label-amplitude shortcut** — drone clips are short, peak-normalized
     (RMS 0.21–0.28); no-drone clips are long, unnormalized (RMS 0.002–0.08). The model
     can learn "loud → drone" from absolute amplitude with zero acoustic content. At
     inference, the gain knob pushes everything into the loud regime → all-ones collapse
     (the v5/v6 signature).

Both bugs vanish when both sides land at the same target RMS regardless of mic gain or
DADS clip type. Wide-gain augmentation (D-01..D-04) survives but its purpose changes:
it now teaches robustness to gain variation around the normalized target, NOT to bridge
a 50–60 dB domain gap.

Purpose: Plans 20-04 and 20-07 fix the trainer's *capability* to learn. This plan fixes
the *signal* the trainer learns from. Without D-34, even a perfectly-trained v7 will
re-collapse at inference because train and live inputs are in different amplitude regimes.

Output: shared `_rms_normalize` helper, both pipelines wired through it, tests assert the
RMS contract holds end-to-end on both sides.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/debug/training-collapse-constant-output.md
@scripts/verify_rms_domain_mismatch.py
@src/acoustic/classification/preprocessing.py
@src/acoustic/training/augmentation.py
@src/acoustic/training/efficientat_trainer.py
@src/acoustic/training/hf_dataset.py
@src/acoustic/training/config.py
@src/acoustic/config.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Shared _rms_normalize helper + unit tests</name>
  <files>
    src/acoustic/classification/preprocessing.py
    tests/unit/test_rms_normalize.py
  </files>
  <read_first>
    src/acoustic/classification/preprocessing.py
  </read_first>
  <behavior>
    Add a small pure function `_rms_normalize(audio, target=0.1, eps=1e-6)` to
    `preprocessing.py` (or a new `src/acoustic/dsp/rms.py` if you prefer to keep
    `preprocessing.py` UI-facing). Behavior:

    - Accepts `np.ndarray` (1-D float32) OR `torch.Tensor` (1-D float32). Returns the
      same type it received.
    - Computes `current_rms = sqrt(mean(audio ** 2))`.
    - If `current_rms < eps`, returns the input unchanged (silence stays silence; no
      divide-by-zero amplification of noise floors).
    - Otherwise multiplies by `target / current_rms` so the output has RMS == target.
    - Does NOT clip; downstream code handles saturation if any (the target=0.1 is
      well below 1.0 so saturation is unlikely).
    - Idempotent: applying it twice yields the same output as applying it once
      (within float32 precision).

    Tests in `tests/unit/test_rms_normalize.py`:
      a) numpy float32 input → output RMS within 1e-5 of target
      b) torch float32 input → output RMS within 1e-5 of target, returned type is torch
      c) silence (all zeros) → output unchanged
      d) sub-eps signal (RMS < eps) → output unchanged
      e) idempotence: `_rms_normalize(_rms_normalize(x))` ≈ `_rms_normalize(x)`
      f) target override: `_rms_normalize(x, target=0.5)` → output RMS within 1e-5 of 0.5
  </behavior>
  <action>
    Step 1 — Add `_rms_normalize` near the top of `src/acoustic/classification/preprocessing.py`,
    above `class RawAudioPreprocessor`:
    ```python
    def _rms_normalize(
        audio,  # np.ndarray | torch.Tensor
        target: float = 0.1,
        eps: float = 1e-6,
    ):
        """Scale waveform so its RMS equals `target`. Silence (RMS<eps) returned as-is.

        Used on BOTH the training dataset path AND RawAudioPreprocessor.process()
        so the model sees the same amplitude distribution at train and inference time.
        See D-34 + .planning/debug/training-collapse-constant-output.md.
        """
        if isinstance(audio, torch.Tensor):
            current_rms = torch.sqrt(torch.mean(audio * audio))
            if current_rms.item() < eps:
                return audio
            return audio * (target / current_rms)
        # numpy path
        current_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if current_rms < eps:
            return audio
        return (audio * (target / current_rms)).astype(audio.dtype)
    ```

    Step 2 — Create `tests/unit/test_rms_normalize.py` with the six cases above.
  </action>
  <verify>
    <automated>pytest tests/unit/test_rms_normalize.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "def _rms_normalize" src/acoustic/classification/preprocessing.py` returns 1 match
    - `pytest tests/unit/test_rms_normalize.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    Shared helper exists with full unit coverage including edge cases.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire RawAudioPreprocessor to RMS-normalize after resample</name>
  <files>
    src/acoustic/classification/preprocessing.py
    src/acoustic/config.py
    tests/unit/test_raw_audio_preprocessor.py
  </files>
  <read_first>
    src/acoustic/classification/preprocessing.py,
    src/acoustic/config.py
  </read_first>
  <behavior>
    - `RawAudioPreprocessor.__init__` gains a new arg `rms_normalize_target: float | None = 0.1`.
      `None` disables normalization (escape hatch); the default enables it.
    - `RawAudioPreprocessor.process()` applies `_rms_normalize(waveform, target=self._rms_target)`
      AFTER resample and AFTER the legacy `input_gain` multiply, but BEFORE the debug dump
      (so the dumped WAVs reflect what the model actually sees).
    - `AcousticSettings.cnn_input_gain` default in `src/acoustic/config.py` is changed
      from its current value (likely 500.0 — verify) to `1.0`. The field is preserved for
      backwards compatibility but is now effectively a no-op when normalization is on.
      Add a deprecation note in the docstring.
    - `AcousticSettings` gains `cnn_rms_normalize_target: float = 0.1`, plumbed into
      `RawAudioPreprocessor` at construction.
    - Existing `RawAudioPreprocessor` tests still pass; new test asserts that for an
      input waveform with RMS in [0.001, 10.0], the output of `process()` always has
      RMS within 1e-3 of 0.1.
  </behavior>
  <action>
    Step 1 — In `src/acoustic/classification/preprocessing.py:124` (the `__init__`),
    add the new arg and store it:
    ```python
    def __init__(
        self,
        target_sr: int = 32000,
        input_gain: float = 1.0,
        rms_normalize_target: float | None = 0.1,
    ) -> None:
        self._target_sr = target_sr
        self._resampler = None
        self._cached_sr = None
        self._input_gain = float(input_gain)
        self._rms_target = rms_normalize_target  # None disables, default 0.1
        # ... rest of existing __init__ unchanged
    ```

    Step 2 — In `process()` (line 148), insert the normalization step AFTER the input_gain
    multiply and BEFORE the debug dump:
    ```python
    def process(self, audio, sr):
        waveform = torch.from_numpy(audio).float()
        if sr != self._target_sr:
            if self._cached_sr != sr:
                self._resampler = torchaudio.transforms.Resample(sr, self._target_sr)
                self._cached_sr = sr
            waveform = self._resampler(waveform)
        if self._input_gain != 1.0:
            waveform = waveform * self._input_gain

        # D-34: RMS normalization (the actual amplitude calibration).
        # Replaces the implicit calibration that input_gain used to do.
        if self._rms_target is not None:
            waveform = _rms_normalize(waveform, target=self._rms_target)

        if self._dump_dir is not None:
            self._dump(waveform)

        return waveform
    ```

    Step 3 — In `src/acoustic/config.py` (`AcousticSettings`), find `cnn_input_gain`,
    change its default to 1.0, add `cnn_rms_normalize_target: float = 0.1` next to it
    with a docstring referencing D-34. Update the docstring of `cnn_input_gain` to note
    it's deprecated in favor of normalization.

    Step 4 — Find every construction site of `RawAudioPreprocessor` (grep
    `RawAudioPreprocessor(`) and pass `rms_normalize_target=settings.cnn_rms_normalize_target`
    where `settings` is in scope. Likely sites: `src/acoustic/main.py`, `src/acoustic/api/`,
    and any tests that construct it directly.

    Step 5 — Update the `RawAudioPreprocessor` class docstring (lines 104-122) to reflect
    the new behavior — drop the "500x" gap explanation, add "RMS-normalized to target=0.1
    on every chunk, regardless of mic gain. See D-34."

    Step 6 — Create `tests/unit/test_raw_audio_preprocessor.py` (or add to existing if
    one exists). New test:
    ```python
    @pytest.mark.parametrize("input_rms", [0.001, 0.01, 0.1, 1.0, 10.0])
    def test_process_normalizes_to_target_rms(input_rms):
        sr = 48000
        n = sr  # 1 second
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(n).astype(np.float32)
        audio *= input_rms / np.sqrt(np.mean(audio ** 2))
        pre = RawAudioPreprocessor(target_sr=32000, input_gain=1.0, rms_normalize_target=0.1)
        out = pre.process(audio, sr).numpy()
        out_rms = float(np.sqrt(np.mean(out ** 2)))
        assert abs(out_rms - 0.1) < 1e-3, f"input_rms={input_rms} → out_rms={out_rms}"
    ```
  </action>
  <verify>
    <automated>pytest tests/unit/test_raw_audio_preprocessor.py tests/unit/test_rms_normalize.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "rms_normalize_target" src/acoustic/classification/preprocessing.py` returns 2+ matches (init + process)
    - `grep -n "_rms_normalize" src/acoustic/classification/preprocessing.py` returns 2+ matches (def + call)
    - `grep -n "cnn_rms_normalize_target" src/acoustic/config.py` returns 1+ matches
    - `grep -n "cnn_input_gain.*=.*500\|cnn_input_gain.*=.*= 500" src/acoustic/config.py` returns 0 matches (default is no longer 500)
    - `pytest tests/unit/test_raw_audio_preprocessor.py -x -q` exits 0 (parametric RMS test passes for all 5 input levels)
    - `pytest tests/unit/test_rms_normalize.py -x -q` exits 0
    - Existing pipeline integration tests still pass: `pytest tests/integration/test_cnn_pipeline.py -x -q`
  </acceptance_criteria>
  <done>
    Live pipeline now lands every CNN input at RMS 0.1 regardless of mic gain. The 500x
    gain hack is gone; the gain knob is harmless legacy.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: RmsNormalize augmentation + wire it as the LAST step in the trainer chain</name>
  <files>
    src/acoustic/training/augmentation.py
    src/acoustic/training/efficientat_trainer.py
    src/acoustic/training/config.py
    tests/unit/training/test_rms_normalize_augmentation.py
  </files>
  <read_first>
    src/acoustic/training/augmentation.py,
    src/acoustic/training/efficientat_trainer.py,
    src/acoustic/training/config.py,
    src/acoustic/classification/preprocessing.py
  </read_first>
  <behavior>
    - New augmentation class `RmsNormalize(target=0.1, eps=1e-6)` in
      `src/acoustic/training/augmentation.py`. It calls the same `_rms_normalize` helper
      from `acoustic.classification.preprocessing` (do NOT duplicate the implementation).
    - It is pickle-safe (no closures, no lambdas) and follows the existing augmentation
      class interface (`__call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray`).
    - `EfficientATTrainingRunner._build_train_augmentation()` (added in Plan 20-04) appends
      `RmsNormalize(target=cfg.rms_normalize_target)` as the LAST step in the chain:
      `WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer → RmsNormalize`.
    - `_build_eval_augmentation()` ALSO appends `RmsNormalize` as the last step. Eval is
      not exempt — the model must see normalized inputs in val/test too, otherwise val
      metrics drift relative to live inference.
    - `TrainingConfig` gains `rms_normalize_target: float = 0.1`.
    - New test asserts:
        a) `RmsNormalize` produces output with RMS ≈ target for non-silent input
        b) `RmsNormalize` returns silence unchanged
        c) The composed train chain ends with `RmsNormalize`
        d) The composed eval chain ends with `RmsNormalize`
  </behavior>
  <action>
    Step 1 — Add to `src/acoustic/training/augmentation.py`:
    ```python
    from acoustic.classification.preprocessing import _rms_normalize

    class RmsNormalize:
        """RMS-normalize a waveform to a fixed target. Pickle-safe.

        Must run AFTER BackgroundNoiseMixer so SNR-mixed signals are normalized
        as a unit, not pre-mix. See D-34 in 20-CONTEXT.md.
        """

        def __init__(self, target: float = 0.1, eps: float = 1e-6) -> None:
            self.target = float(target)
            self.eps = float(eps)

        def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
            return _rms_normalize(audio, target=self.target, eps=self.eps)
    ```

    Step 2 — Add to `TrainingConfig`:
    ```python
    rms_normalize_target: float = Field(
        default=0.1,
        description="Target RMS for per-sample normalization. Applied as the LAST step "
                    "in the augmentation chain on both training and eval splits, and "
                    "on RawAudioPreprocessor.process() at inference. See D-34.",
    )
    ```

    Step 3 — In `efficientat_trainer.py`, update `_build_train_augmentation()` (added in
    Plan 20-04) — after the BackgroundNoiseMixer block, append:
    ```python
    # Stage 5: RMS normalization — LAST in the chain (D-34).
    # Kills the train/inference domain shift AND the DADS amplitude shortcut.
    augs.append(RmsNormalize(target=cfg.rms_normalize_target))
    ```
    Add the same line at the end of `_build_eval_augmentation()`. Remember the existing
    `_build_eval_augmentation` from Plan 20-04 may return `None` if no other augmentations
    are configured — in that case it must still return a `ComposedAugmentation([RmsNormalize(...)])`
    so eval normalization runs unconditionally.

    Step 4 — Import `RmsNormalize` in `efficientat_trainer.py`:
    ```python
    from acoustic.training.augmentation import (
        WideGainAugmentation,
        RoomIRAugmentation,
        AudiomentationsAugmentation,
        BackgroundNoiseMixer,
        ComposedAugmentation,
        RmsNormalize,
    )
    ```

    Step 5 — Create `tests/unit/training/test_rms_normalize_augmentation.py`:
    - Direct test of `RmsNormalize.__call__` on synthetic 1-second sine waves at
      varying amplitudes.
    - Test that constructs `_build_train_augmentation()` from a default config and
      asserts the LAST element of the returned `ComposedAugmentation`'s aug list is
      a `RmsNormalize` instance with `target == 0.1`.
    - Same assertion for `_build_eval_augmentation()`.
  </action>
  <verify>
    <automated>pytest tests/unit/training/test_rms_normalize_augmentation.py tests/unit/training/test_trainer_augmentation_order.py tests/unit/test_efficientat_training.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "class RmsNormalize" src/acoustic/training/augmentation.py` returns 1 match
    - `grep -n "from acoustic.classification.preprocessing import _rms_normalize" src/acoustic/training/augmentation.py` returns 1 match
    - `grep -n "RmsNormalize(target=cfg.rms_normalize_target)" src/acoustic/training/efficientat_trainer.py` returns 2 matches (train + eval builders)
    - `grep -n "rms_normalize_target" src/acoustic/training/config.py` returns 1+ matches
    - `pytest tests/unit/training/test_rms_normalize_augmentation.py -x -q` exits 0
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    The training chain ends with the same RMS normalization that the live pipeline starts
    with. Both sides land at RMS=0.1.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: End-to-end RMS contract test (train↔inference parity)</name>
  <files>
    tests/integration/test_rms_contract_train_inference.py
  </files>
  <read_first>
    src/acoustic/classification/preprocessing.py,
    src/acoustic/training/augmentation.py,
    src/acoustic/training/efficientat_trainer.py
  </read_first>
  <behavior>
    A single integration test that exercises the full chain on both sides and asserts
    they land in the same regime. This is the regression gate for D-34 — if it ever
    fails, the train/inference contract has drifted.

    Generate three synthetic 1-second waveforms at very different RMS levels (0.001,
    0.18, 9.3 — chosen to mirror the empirical findings: silence-ish, DADS-typical,
    live-UMA-16-typical-after-cnn-input-gain-was-still-on). Run each through:
      a) `RawAudioPreprocessor(rms_normalize_target=0.1).process(audio, sr=48000)`
      b) The trainer's `_build_train_augmentation()` ComposedAugmentation
      c) The trainer's `_build_eval_augmentation()` ComposedAugmentation

    For each path, compute output RMS. Assert all three outputs are within 1e-3 of 0.1
    EXCEPT the silence case, which should remain at ~0.001 (below the eps threshold).

    Skip background noise mixer in this test (use a config with
    `noise_augmentation_enabled=False`) so the test doesn't depend on noise dirs being
    populated. The RMS contract is independent of noise mixing.
  </behavior>
  <action>
    Create `tests/integration/test_rms_contract_train_inference.py`. Use existing test
    fixtures or build a minimal `TrainingConfig` from scratch with:
    `noise_augmentation_enabled=False, rir_enabled=False, use_audiomentations=False, wide_gain_db=0`.

    The test should fail loudly if any future change reintroduces a domain shift.
  </action>
  <verify>
    <automated>pytest tests/integration/test_rms_contract_train_inference.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `tests/integration/test_rms_contract_train_inference.py` exists
    - `pytest tests/integration/test_rms_contract_train_inference.py -x -q` exits 0
    - The test asserts both train and inference paths produce RMS within 1e-3 of 0.1 for non-silent inputs at all three reference amplitudes (0.18, 9.3, plus a fourth ~1e-3 silence-floor case kept unchanged)
  </acceptance_criteria>
  <done>
    Train↔inference RMS parity is now a CI-enforced contract.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Live audio → preprocessor | Live UMA-16 input is untrusted in amplitude only (clipping, DC offset). RMS normalization saturates the output if input is malformed but does not introduce a vulnerability. |
| Training audio → dataset | HF DADS samples are trusted; normalization just unifies amplitude. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-08-01 | Tampering (silent contract drift) | Train/inference RMS parity | mitigate | Task 4 integration test acts as a regression gate. |
| T-20-08-02 | Information Disclosure (legacy gain still in some path) | RawAudioPreprocessor input_gain | accept | Field is preserved for backwards compat but defaults to 1.0; normalization makes it irrelevant. Operators who set it manually get a warning logged. |
| T-20-08-03 | Denial of Service (silence amplification) | _rms_normalize on near-silence | mitigate | eps=1e-6 short-circuit; sub-eps signals returned unchanged so noise floors are not amplified. |
</threat_model>

<verification>
- All four task automated checks pass
- After this plan + Plan 20-07 + Plan 20-04, a Vertex v7 run should produce a model
  whose val confusion matrix is non-degenerate within the first few epochs of Stage 1
- Live pipeline output of `RawAudioPreprocessor.process()` should have RMS ≈ 0.1
  regardless of mic gain — quick smoke test via the existing debug dump
</verification>

<success_criteria>
- `pytest tests/unit/test_rms_normalize.py tests/unit/test_raw_audio_preprocessor.py tests/unit/training/test_rms_normalize_augmentation.py tests/integration/test_rms_contract_train_inference.py tests/unit/test_efficientat_training.py tests/integration/test_cnn_pipeline.py -x -q` exits 0
- Empirical follow-up: re-run `scripts/verify_rms_domain_mismatch.py` after the changes
  and confirm DADS-vs-live ratio drops from 52x to ~1x
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-08-SUMMARY.md`
</output>
