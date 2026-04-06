---
phase: 20
plan: 07
type: execute
wave: 3
depends_on:
  - "20-00"
  - "20-04"
files_modified:
  - src/acoustic/training/efficientat_trainer.py
  - src/acoustic/training/config.py
autonomous: true
requirements:
  - D-30
  - D-31
  - D-32
  - D-33
must_haves:
  truths:
    - "mel_train SpecAugment uses freqm/timem from config (defaults specaug_freq_mask=8, specaug_time_mask=10) — NOT the legacy 48/192"
    - "Loss is constructed via build_loss_function(cfg) — focal when cfg.loss_function == 'focal', BCE with optional pos_weight otherwise"
    - "Per-epoch save hook refuses to save the checkpoint when min(tp, tn) == 0 OR val_accuracy < 0.55, and logs a warning"
    - "Stage 1 unfreezes ONLY the new final binary head (Linear(1280, 1)) — not the full pretrained classifier MLP"
    - "Stage 2 still unfreezes the rest of the classifier (existing behavior preserved)"
    - "Existing test_efficientat_training.py tests still pass"
  artifacts:
    - path: src/acoustic/training/efficientat_trainer.py
      provides: "Trainer correctness fixes for SpecAugment, loss factory, save gate, stage 1 scope"
      contains: "build_loss_function"
    - path: src/acoustic/training/config.py
      provides: "specaug_freq_mask, specaug_time_mask, save_gate_min_accuracy config fields"
      contains: "specaug_freq_mask"
  key_links:
    - from: src/acoustic/training/efficientat_trainer.py
      to: src/acoustic/training/losses.py
      via: "criterion = build_loss_function(cfg)"
      pattern: "build_loss_function"
    - from: src/acoustic/training/efficientat_trainer.py
      to: src/acoustic/training/config.py
      via: "AugmentMelSTFT(..., freqm=cfg.specaug_freq_mask, timem=cfg.specaug_time_mask)"
      pattern: "specaug_freq_mask"
---

<objective>
Apply the four trainer-correctness fixes (D-30..D-33) identified by the training-collapse
diagnosis (`.planning/debug/training-collapse-constant-output.md`). Without these fixes,
the Phase 20 augmentation chain wired by Plan 20-04 cannot prevent v7 from collapsing in
the same way v3/v5/v6 did. This plan addresses:

  1. SpecAugment time/freq mask params scaled to the actual ~100-frame input (D-30)
  2. Loss factory wiring so `loss_function="focal"` actually takes effect (D-31)
  3. Behavioral checkpoint save gate that refuses degenerate models (D-32)
  4. Stage 1 unfreezing narrowed to the new final binary head only (D-33)

Purpose: every other Phase 20 plan assumes the trainer is *capable* of producing a non-
degenerate model. The diagnosis proved it currently is not. This is the smallest set of
edits that restores that capability before v7 is submitted to Vertex.

Output: Modified trainer + config; new unit tests for each fix; existing tests still green.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/debug/training-collapse-constant-output.md
@src/acoustic/training/efficientat_trainer.py
@src/acoustic/training/losses.py
@src/acoustic/training/config.py
@src/acoustic/training/trainer.py
@tests/unit/test_efficientat_training.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Scale SpecAugment masks to actual input dim (D-30)</name>
  <files>
    src/acoustic/training/config.py
    src/acoustic/training/efficientat_trainer.py
    tests/unit/training/test_specaug_scaling.py
  </files>
  <read_first>
    src/acoustic/training/efficientat_trainer.py,
    src/acoustic/training/config.py,
    src/acoustic/classification/efficientat/preprocess.py,
    src/acoustic/classification/efficientat/config.py,
    .planning/debug/training-collapse-constant-output.md
  </read_first>
  <behavior>
    - `TrainingConfig` exposes two new fields with safe defaults:
      `specaug_freq_mask: int = 8` and `specaug_time_mask: int = 10`.
    - `efficientat_trainer.py:346-350` constructs `mel_train` using
      `freqm=cfg.specaug_freq_mask, timem=cfg.specaug_time_mask`.
    - `mel_eval` (line 351) is unchanged (`freqm=0, timem=0`).
    - A new test `test_specaug_scaling.py` constructs the trainer with default config and
      asserts that `mel_train.freqm == 8` and `mel_train.timem == 10`. It also asserts that
      `mel_train.timem` is strictly less than the configured `input_dim_t` (≈100).
    - The legacy values 48/192 are no longer hardcoded anywhere on the EfficientAT path.
  </behavior>
  <action>
    Step 1 — Add to `TrainingConfig` (alphabetically near other `specaug_*` or augmentation
    fields, with pydantic `Field(..., description=...)` matching the existing style):
    ```python
    specaug_freq_mask: int = Field(
        default=8,
        description="SpecAugment frequency mask width (mels). Must be ≤ n_mels // 8.",
    )
    specaug_time_mask: int = Field(
        default=10,
        description="SpecAugment time mask width (frames). Must be ≤ input_dim_t // 10. "
                    "Legacy value 192 caused full-axis masking on 100-frame inputs and "
                    "drove the v3/v5/v6 constant-output collapse — see "
                    ".planning/debug/training-collapse-constant-output.md",
    )
    ```

    Step 2 — In `efficientat_trainer.py:346-350`, replace the literal masks:
    ```python
    mel_train = AugmentMelSTFT(
        n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
        win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
        n_fft=mel_cfg.n_fft,
        freqm=cfg.specaug_freq_mask,
        timem=cfg.specaug_time_mask,
    )
    ```
    Leave `mel_eval` (line 351) untouched — eval correctly uses 0/0.

    Step 3 — Create `tests/unit/training/test_specaug_scaling.py`. Use the existing
    runner-construction fixture from `test_efficientat_training.py` if one exists; otherwise
    construct `AugmentMelSTFT(..., freqm=cfg.specaug_freq_mask, timem=cfg.specaug_time_mask)`
    directly and assert the attributes. Two assertions:
      - default config: `freqm == 8 and timem == 10`
      - sanity bound: `timem < EfficientATMelConfig().input_dim_t // 5` (i.e. ≤20 for 100 frames)
  </action>
  <verify>
    <automated>pytest tests/unit/training/test_specaug_scaling.py tests/unit/test_efficientat_training.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "freqm=cfg.specaug_freq_mask" src/acoustic/training/efficientat_trainer.py` returns 1 match
    - `grep -n "timem=cfg.specaug_time_mask" src/acoustic/training/efficientat_trainer.py` returns 1 match
    - `grep -n "freqm=48\|timem=192" src/acoustic/training/efficientat_trainer.py` returns 0 matches
    - `grep -n "specaug_freq_mask\|specaug_time_mask" src/acoustic/training/config.py` returns 2+ matches
    - `pytest tests/unit/training/test_specaug_scaling.py -x -q` exits 0
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0 (no regression)
  </acceptance_criteria>
  <done>
    SpecAugment masks are config-driven and proportional to input length.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire build_loss_function() into the trainer (D-31)</name>
  <files>
    src/acoustic/training/efficientat_trainer.py
    tests/unit/training/test_trainer_loss_factory.py
  </files>
  <read_first>
    src/acoustic/training/efficientat_trainer.py,
    src/acoustic/training/losses.py,
    src/acoustic/training/config.py
  </read_first>
  <behavior>
    - `efficientat_trainer.py:357` no longer constructs `nn.BCEWithLogitsLoss()` directly.
      Instead it calls `criterion = build_loss_function(cfg)`.
    - `build_loss_function` already returns a focal loss when `cfg.loss_function == "focal"`
      (using `cfg.focal_alpha`, `cfg.focal_gamma`) and BCE otherwise (with optional
      `cfg.bce_pos_weight`). Reuse it as-is — do not duplicate the logic.
    - The factory's return must be compatible with the existing training loop's call
      `criterion(logits, target)` where logits/target are the same shape they are today.
      If `build_loss_function` returns something with a different signature, ADAPT THE
      TRAINER call site, not the factory.
    - New test asserts:
        a) with `cfg.loss_function == "focal"`, the constructed criterion is the focal class
        b) with `cfg.loss_function == "bce"` (default), it is BCE-flavored
        c) `cfg.bce_pos_weight` is honored for the BCE branch
  </behavior>
  <action>
    Step 1 — In `efficientat_trainer.py`, add the import near other training imports:
    ```python
    from acoustic.training.losses import build_loss_function
    ```

    Step 2 — At line 357, replace:
    ```python
    criterion = nn.BCEWithLogitsLoss()
    ```
    with:
    ```python
    criterion = build_loss_function(cfg)
    ```

    Step 3 — If `build_loss_function`'s return type differs from `nn.BCEWithLogitsLoss`
    (e.g. expects `(logits, target, weight=None)` instead of `(logits, target)`), inspect
    the call sites of `criterion(...)` in the training and validation loops and adapt
    them. DO NOT change the factory.

    Step 4 — Create `tests/unit/training/test_trainer_loss_factory.py`:
    ```python
    from acoustic.training.config import TrainingConfig
    from acoustic.training.losses import build_loss_function
    import torch.nn as nn

    def test_focal_selected_when_configured():
        cfg = TrainingConfig(loss_function="focal", focal_alpha=0.25, focal_gamma=2.0)
        loss = build_loss_function(cfg)
        # exact class name depends on losses.py — assert "focal" in repr or class name
        assert "focal" in type(loss).__name__.lower()

    def test_bce_default():
        cfg = TrainingConfig()  # defaults
        loss = build_loss_function(cfg)
        assert isinstance(loss, (nn.BCEWithLogitsLoss,)) or "bce" in type(loss).__name__.lower()

    def test_bce_pos_weight_honored():
        cfg = TrainingConfig(loss_function="bce", bce_pos_weight=2.5)
        loss = build_loss_function(cfg)
        # introspect — exact attribute depends on factory implementation
        # at minimum assert no exception and loss is callable
        import torch
        out = loss(torch.zeros(4, 1), torch.zeros(4, 1))
        assert out is not None
    ```
    Adjust assertions to match the actual class names exposed by `losses.py` after reading it.
  </action>
  <verify>
    <automated>pytest tests/unit/training/test_trainer_loss_factory.py tests/unit/test_efficientat_training.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "build_loss_function(cfg)" src/acoustic/training/efficientat_trainer.py` returns 1 match
    - `grep -n "nn.BCEWithLogitsLoss()" src/acoustic/training/efficientat_trainer.py` returns 0 matches
    - `pytest tests/unit/training/test_trainer_loss_factory.py -x -q` exits 0
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    Loss is config-driven; `loss_function="focal"` from env vars takes effect at training time.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Behavioral save gate — refuse degenerate checkpoints (D-32)</name>
  <files>
    src/acoustic/training/config.py
    src/acoustic/training/efficientat_trainer.py
    tests/unit/training/test_save_gate.py
  </files>
  <read_first>
    src/acoustic/training/efficientat_trainer.py,
    src/acoustic/training/trainer.py
  </read_first>
  <behavior>
    - New `TrainingConfig` field: `save_gate_min_accuracy: float = 0.55`.
    - In the per-epoch save block (the existing path that writes `best_model.pt`), the
      runner consults the val confusion matrix already computed at line ~489. If
      `min(tp, tn) == 0` OR `val_accuracy < cfg.save_gate_min_accuracy`, the checkpoint
      is NOT saved and a `WARNING` is logged via the existing `_logger` instance with
      message "save gate blocked: tp=X tn=Y val_acc=Z (degenerate output)".
    - The save gate does NOT affect `EarlyStopping` patience tracking (it can still terminate
      training based on val_loss). It is a side guard, not a control loop.
    - The existing improvement-detection (val_loss minimum) still runs first; the gate is
      a second condition AND-ed with it.
    - New test constructs a fake epoch state with tp=0, tn=100 (all-zero collapse) and asserts
      the save hook does not write the file. Second test with tp=80, tn=70 (healthy) asserts
      the file IS written.
  </behavior>
  <action>
    Step 1 — Add to `TrainingConfig`:
    ```python
    save_gate_min_accuracy: float = Field(
        default=0.55,
        description="Refuse to save checkpoints below this val accuracy. Guards against "
                    "degenerate constant-output models that achieve stable val_loss but "
                    "zero TP or zero TN. See .planning/debug/training-collapse-constant-output.md",
    )
    ```

    Step 2 — Locate the per-epoch save site in `efficientat_trainer.py` (search for
    `best_model` or `torch.save`). The val confusion matrix `(tp, fp, tn, fn)` is already
    computed in the val loop at line ~489 for the progress callback. Plumb those four
    integers down to the save site (or recompute them — they're cheap).

    Step 3 — Wrap the existing save call with a gate:
    ```python
    val_total = tp + tn + fp + fn
    val_accuracy = (tp + tn) / val_total if val_total > 0 else 0.0
    save_gate_ok = (min(tp, tn) > 0) and (val_accuracy >= cfg.save_gate_min_accuracy)

    if val_loss_improved and save_gate_ok:
        torch.save(...)  # existing save call
    elif val_loss_improved and not save_gate_ok:
        self._logger.warning(
            "save gate blocked: tp=%d tn=%d val_acc=%.3f (degenerate output)",
            tp, tn, val_accuracy,
        )
    ```

    Step 4 — Create `tests/unit/training/test_save_gate.py` with two cases. Mock or use a
    helper to invoke the save logic with synthetic confusion matrix values; assert
    `torch.save` is called for healthy and NOT called for degenerate.
  </action>
  <verify>
    <automated>pytest tests/unit/training/test_save_gate.py tests/unit/test_efficientat_training.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "save_gate_min_accuracy" src/acoustic/training/config.py` returns 1+ matches
    - `grep -n "save_gate_min_accuracy\|save gate blocked" src/acoustic/training/efficientat_trainer.py` returns 2+ matches
    - `pytest tests/unit/training/test_save_gate.py -x -q` exits 0
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    Constant-output checkpoints can no longer slip through to disk silently.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: Narrow Stage 1 unfreezing to final head only (D-33)</name>
  <files>
    src/acoustic/training/efficientat_trainer.py
    tests/unit/training/test_stage1_unfreeze_scope.py
  </files>
  <read_first>
    src/acoustic/training/efficientat_trainer.py,
    .planning/debug/training-collapse-constant-output.md
  </read_first>
  <behavior>
    - `_setup_stage1` (around `efficientat_trainer.py:137-142`) unfreezes ONLY the final
      `Linear(1280, 1)` binary head, not the entire pretrained classifier MLP.
    - `_setup_stage2` is UNCHANGED. It continues to unfreeze the rest of the classifier as
      it does today.
    - The final-head reference is whatever the binary head replacement at line ~342
      sets — typically `model.classifier[-1]` after the MN.classifier replacement, or a
      named attribute. Use the same reference here for consistency.
    - New test asserts that after `_setup_stage1`:
        a) the final binary `Linear(1280, 1)` weights have `requires_grad == True`
        b) the preceding `Linear(1280, 1280)` (if present in the pretrained head) has
           `requires_grad == False`
        c) all backbone (non-classifier) params remain frozen
  </behavior>
  <action>
    Step 1 — Read `_setup_stage1` (around line 137-142). Currently:
    ```python
    for p in model.classifier.parameters():
        p.requires_grad = True
    ```
    Change to:
    ```python
    # D-33: Stage 1 only unfreezes the new final binary head; the pretrained
    # Linear(1280, 1280) preceding it stays frozen until Stage 2. This prevents
    # Adam at 1e-3 from collapsing the head over masked-input batches.
    final_head = model.classifier[-1]  # adjust to actual attribute name as needed
    for p in final_head.parameters():
        p.requires_grad = True
    ```
    If the binary head replacement at line ~342 uses a different attribute path (e.g.
    `model.binary_head`), use that instead. The key invariant is "exactly the params of
    the new Linear(1280, 1)".

    Step 2 — Confirm `_setup_stage2` still unfreezes `model.classifier.parameters()` in
    full. If yes, no change there. If it instead incrementally unfreezes from where stage 1
    left off, verify the chain still ends in "all classifier params trainable by stage 2".

    Step 3 — Create `tests/unit/training/test_stage1_unfreeze_scope.py`:
    ```python
    from acoustic.classification.efficientat.model import build_efficientat_mn10  # or actual builder
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner
    import torch.nn as nn

    def test_stage1_unfreezes_only_final_head():
        runner = EfficientATTrainingRunner(...)  # use existing test fixture
        model = runner._build_model()  # or however the model is constructed
        runner._setup_stage1(model)

        # Final head trainable
        final_head = model.classifier[-1]
        assert all(p.requires_grad for p in final_head.parameters())

        # Preceding classifier layer frozen
        # find the prior Linear(1280, 1280) in model.classifier
        prior_linears = [m for m in model.classifier if isinstance(m, nn.Linear)][:-1]
        for layer in prior_linears:
            assert not any(p.requires_grad for p in layer.parameters()), \
                "Stage 1 must NOT unfreeze the pretrained Linear(1280, 1280) head"

        # Backbone fully frozen
        backbone_params = [
            p for name, p in model.named_parameters()
            if not name.startswith("classifier")
        ]
        assert not any(p.requires_grad for p in backbone_params)
    ```
    Adjust the model construction call to match whatever fixture the existing
    `test_efficientat_training.py` uses.
  </action>
  <verify>
    <automated>pytest tests/unit/training/test_stage1_unfreeze_scope.py tests/unit/test_efficientat_training.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "model.classifier.parameters" src/acoustic/training/efficientat_trainer.py` shows the
      old usage is gone from `_setup_stage1` (it may still appear in `_setup_stage2`)
    - `grep -n "final_head\|classifier\[-1\]" src/acoustic/training/efficientat_trainer.py` returns matches
    - `pytest tests/unit/training/test_stage1_unfreeze_scope.py -x -q` exits 0
    - `pytest tests/unit/test_efficientat_training.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    Stage 1 trains only the final binary head; the pretrained MLP classifier stays frozen
    until Stage 2.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| TrainingConfig env vars → trainer | New fields `specaug_freq_mask`, `specaug_time_mask`, `save_gate_min_accuracy` come from pydantic settings; no untrusted source. |
| Loss factory → training loop | `build_loss_function` returns a callable; trainer must not assume `nn.BCEWithLogitsLoss` interface beyond `(logits, target) -> loss`. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-07-01 | Information Disclosure (silent regression) | save gate | mitigate | Gate logs WARNING when blocking; operator sees the degenerate state in logs instead of getting a bad checkpoint shipped silently. |
| T-20-07-02 | Tampering (config override) | specaug_*_mask defaults | accept | Operators can still override to harmful values via env vars; documented in field description. Out of scope to enforce a hard upper bound. |
| T-20-07-03 | Denial of Service (training never converges) | stage1 narrow unfreeze | mitigate | Stage 2 still unfreezes the full classifier within 10 epochs, so worst case is slower convergence, not non-convergence. Existing test_efficientat_training.py covers convergence on a tiny synthetic dataset. |
</threat_model>

<verification>
- All four sub-tests pass
- `tests/unit/test_efficientat_training.py` still passes (no regression on the existing
  end-to-end training smoke test)
- Manual: after this plan + Plan 20-04, a Vertex v7 run with `loss_function=focal` should
  produce a non-degenerate val confusion matrix in the first few epochs of Stage 1; if it
  doesn't, the save gate (D-32) will block the checkpoint write and flag in logs.
</verification>

<success_criteria>
- `pytest tests/unit/training/test_specaug_scaling.py tests/unit/training/test_trainer_loss_factory.py tests/unit/training/test_save_gate.py tests/unit/training/test_stage1_unfreeze_scope.py tests/unit/test_efficientat_training.py -x -q` exits 0
- All grep acceptance checks across the four tasks pass
- The four diagnosis bugs (PRIMARY-A, PRIMARY-C, CONTRIBUTING-D, CONTRIBUTING-F) are
  closed at their root, not patched around
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-07-SUMMARY.md`
</output>
