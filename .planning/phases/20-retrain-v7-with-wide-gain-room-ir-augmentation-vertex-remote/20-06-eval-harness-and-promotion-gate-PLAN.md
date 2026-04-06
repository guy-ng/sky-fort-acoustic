---
phase: 20
plan: 06
type: execute
wave: 5
depends_on:
  - "20-05"
files_modified:
  - src/acoustic/evaluation/uma16_eval.py
  - src/acoustic/evaluation/evaluator.py
  - src/acoustic/evaluation/promotion.py
  - scripts/promote_v7.py
autonomous: false
requirements:
  - D-26
  - D-27
  - D-28
  - D-29
must_haves:
  truths:
    - "Eval harness loads the UMA-16 real-capture eval set from data/eval/uma16_real/labels.json"
    - "Eval harness emits confusion matrix + ROC curve (sklearn.metrics.roc_curve) per D-28"
    - "promote_v7_if_gates_pass(dads_acc, real_tpr, real_fpr) returns True ONLY if dads_acc >= 0.95 AND real_tpr >= 0.80 AND real_fpr <= 0.05"
    - "Promotion script copies models/efficientat_mn10_v7.pt to models/efficientat_mn10.pt ONLY when all three conditions hold"
    - "Promotion script verifies the source checkpoint sha256 matches the value recorded in 20-05-SUMMARY.md before copying"
    - "v7 has been evaluated against both DADS test split (D-26) and real-capture eval set (D-27)"
  artifacts:
    - path: src/acoustic/evaluation/uma16_eval.py
      provides: "load_uma16_eval_set + per-segment scoring against trained classifier"
      contains: "def load_uma16_eval_set"
    - path: src/acoustic/evaluation/evaluator.py
      provides: "ROC curve addition to existing Evaluator output"
      contains: "roc_curve"
    - path: src/acoustic/evaluation/promotion.py
      provides: "promote_v7_if_gates_pass(dads_acc, real_tpr, real_fpr) -> bool gate"
      contains: "def promote_v7_if_gates_pass"
    - path: scripts/promote_v7.py
      provides: "CLI entry: runs both evals, applies gate, optionally cp v7→default"
      contains: "promote_v7_if_gates_pass"
  key_links:
    - from: src/acoustic/evaluation/promotion.py
      to: models/
      via: "shutil.copy2(efficientat_mn10_v7.pt, efficientat_mn10.pt) gated by both metrics"
      pattern: "efficientat_mn10.pt"
    - from: src/acoustic/evaluation/uma16_eval.py
      to: data/eval/uma16_real/labels.json
      via: "json.loads → list[(wav_path, label_int)]"
      pattern: "labels.json"
---

<objective>
Extend the Phase 9 evaluator to consume the real-capture UMA-16 eval set produced in Wave 0
(D-27), emit a ROC curve alongside the existing confusion matrix (D-28), and gate v7 promotion
on both D-26 (DADS test ≥0.95 acc) and D-27 (TPR ≥0.80, FPR ≤0.05) per D-29. The promotion gate
is the END of Phase 20 — it either copies models/efficientat_mn10_v7.pt → models/efficientat_mn10.pt
or leaves v6 as default and the orchestrator routes to gap closure.

Purpose: Without the eval harness extension and the promotion gate, the trained v7 checkpoint is
just a file on disk. This plan turns the trained artifact into a deployment decision per the rules
the user locked in CONTEXT.md.

Output: New uma16_eval.py + promotion.py modules, evaluator.py extended with ROC, promote_v7.py
CLI, manual checkpoint to actually run the gate against the trained v7 checkpoint.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@src/acoustic/evaluation/evaluator.py
@tests/unit/test_evaluator.py
@tests/unit/test_promotion_gate.py

<interfaces>
After plan 05:
- models/efficientat_mn10_v7.pt exists locally
- 20-05-SUMMARY.md records its sha256
- data/eval/uma16_real/labels.json exists with ≥5 min drone + ≥15 min ambient labeled segments
- Existing Evaluator (Phase 9) exposes evaluate_classifier(model_path, dataset) -> EvaluationResult with confusion matrix, precision/recall/F1, distribution stats
- sklearn.metrics.roc_curve(y_true, y_score) is the canonical ROC implementation
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: UMA-16 eval loader + Evaluator ROC extension + promotion gate module</name>
  <files>
    src/acoustic/evaluation/uma16_eval.py,
    src/acoustic/evaluation/evaluator.py,
    src/acoustic/evaluation/promotion.py
  </files>
  <read_first>
    src/acoustic/evaluation/evaluator.py,
    src/acoustic/evaluation/__init__.py,
    tests/unit/test_evaluator.py,
    tests/unit/test_promotion_gate.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    After this task:
    - `from acoustic.evaluation.uma16_eval import load_uma16_eval_set` works
    - `load_uma16_eval_set(eval_dir: Path) -> list[tuple[Path, int]]` reads labels.json and returns [(wav_path, label_int), ...] where label is 1=drone, 0=no_drone
    - Evaluator emits a `roc_curve` field on EvaluationResult populated from `sklearn.metrics.roc_curve`
    - `from acoustic.evaluation.promotion import promote_v7_if_gates_pass` works
    - `promote_v7_if_gates_pass(dads_acc, real_tpr, real_fpr, source_path, target_path, expected_sha256=None) -> bool`:
        * Returns False if any threshold fails (0.95 / 0.80 / 0.05)
        * Returns False if source_path doesn't exist
        * If expected_sha256 provided and computed sha256 mismatches, returns False and logs the mismatch
        * If all checks pass, copies source_path → target_path with shutil.copy2 and returns True
    - All Wave 0 promotion + evaluator tests turn GREEN
  </behavior>
  <action>
    Step 1 — Create src/acoustic/evaluation/uma16_eval.py:
    ```python
    """UMA-16 real-capture eval loader (Phase 20 D-27)."""
    from __future__ import annotations
    import json
    from pathlib import Path

    def load_uma16_eval_set(eval_dir: Path) -> list[tuple[Path, int]]:
        """Load the real-capture eval set from labels.json.

        Expected layout:
            data/eval/uma16_real/
                labels.json   # [{"file": "clip_001.wav", "label": "drone", "start_s": 0, "end_s": 5}, ...]
                clip_001.wav

        Returns list of (wav_path, label_int) where label is 1=drone, 0=no_drone.
        """
        labels_file = Path(eval_dir) / "labels.json"
        if not labels_file.exists():
            raise FileNotFoundError(f"D-27 eval set missing: {labels_file}")
        entries = json.loads(labels_file.read_text())
        out: list[tuple[Path, int]] = []
        for entry in entries:
            path = Path(eval_dir) / entry["file"]
            label = 1 if entry["label"] == "drone" else 0
            out.append((path, label))
        return out
    ```

    Step 2 — Open src/acoustic/evaluation/evaluator.py. Find the EvaluationResult dataclass /
    pydantic model and add an optional field:
    ```python
    roc_curve: list[tuple[float, float, float]] | None = None  # (threshold, fpr, tpr)
    ```
    Then in the `evaluate_classifier` (or equivalent) method, after computing per-segment scores
    and ground-truth labels, add:
    ```python
    try:
        from sklearn.metrics import roc_curve as _sk_roc
        fpr, tpr, thresholds = _sk_roc(np.asarray(all_labels), np.asarray(all_scores))
        result.roc_curve = list(zip(thresholds.tolist(), fpr.tolist(), tpr.tolist()))
    except ImportError:
        result.roc_curve = None
    ```
    If sklearn isn't already in requirements.txt, add it.

    Step 3 — Create src/acoustic/evaluation/promotion.py:
    ```python
    """Phase 20 v7 promotion gate (D-29).

    v7 → models/efficientat_mn10.pt ONLY if BOTH:
      - DADS test accuracy >= 0.95 (D-26)
      - Real-capture TPR >= 0.80 AND FPR <= 0.05 (D-27)
    """
    from __future__ import annotations
    import hashlib
    import logging
    import shutil
    from pathlib import Path

    DADS_ACC_MIN = 0.95
    REAL_TPR_MIN = 0.80
    REAL_FPR_MAX = 0.05

    log = logging.getLogger(__name__)

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def promote_v7_if_gates_pass(
        dads_acc: float,
        real_tpr: float,
        real_fpr: float,
        source_path: Path = Path("models/efficientat_mn10_v7.pt"),
        target_path: Path = Path("models/efficientat_mn10.pt"),
        expected_sha256: str | None = None,
    ) -> bool:
        """Apply the D-29 promotion rule and copy v7 → default if it passes."""
        source_path = Path(source_path)
        target_path = Path(target_path)
        if not source_path.exists():
            log.error("v7 checkpoint missing at %s", source_path)
            return False
        if dads_acc < DADS_ACC_MIN:
            log.warning("D-26 FAIL: dads_acc=%.4f < %.2f", dads_acc, DADS_ACC_MIN)
            return False
        if real_tpr < REAL_TPR_MIN:
            log.warning("D-27 FAIL: real_tpr=%.4f < %.2f", real_tpr, REAL_TPR_MIN)
            return False
        if real_fpr > REAL_FPR_MAX:
            log.warning("D-27 FAIL: real_fpr=%.4f > %.2f", real_fpr, REAL_FPR_MAX)
            return False
        if expected_sha256:
            actual = _sha256(source_path)
            if actual != expected_sha256:
                log.error("v7 sha256 mismatch: expected %s got %s", expected_sha256, actual)
                return False
        shutil.copy2(source_path, target_path)
        log.info(
            "v7 PROMOTED → %s (dads_acc=%.4f real_tpr=%.4f real_fpr=%.4f)",
            target_path, dads_acc, real_tpr, real_fpr,
        )
        return True
    ```

    Step 4 — Ensure module-level imports work: add to src/acoustic/evaluation/__init__.py:
    ```python
    from .uma16_eval import load_uma16_eval_set
    from .promotion import promote_v7_if_gates_pass, DADS_ACC_MIN, REAL_TPR_MIN, REAL_FPR_MAX
    ```
  </action>
  <verify>
    <automated>pytest tests/unit/test_promotion_gate.py tests/unit/test_evaluator.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - File src/acoustic/evaluation/uma16_eval.py exists with `def load_uma16_eval_set`
    - File src/acoustic/evaluation/promotion.py exists with `def promote_v7_if_gates_pass`
    - `grep -n "DADS_ACC_MIN = 0.95" src/acoustic/evaluation/promotion.py` returns one match
    - `grep -n "REAL_TPR_MIN = 0.80" src/acoustic/evaluation/promotion.py` returns one match
    - `grep -n "REAL_FPR_MAX = 0.05" src/acoustic/evaluation/promotion.py` returns one match
    - `grep -n "roc_curve" src/acoustic/evaluation/evaluator.py` returns at least one match in the class definition
    - `grep -n "sha256" src/acoustic/evaluation/promotion.py` confirms supply-chain check
    - `pytest tests/unit/test_promotion_gate.py -x -q` exits 0 (all four threshold tests + checkpoint copy test GREEN)
    - `pytest tests/unit/test_evaluator.py -x -q` exits 0 (no v6 regression + ROC test passes)
  </acceptance_criteria>
  <done>
    UMA-16 eval loader, ROC extension, promotion gate, sha256 verification all in place; promotion
    tests GREEN; existing evaluator tests still GREEN.
  </done>
</task>

<task type="auto">
  <name>Task 2: promote_v7.py CLI orchestrator</name>
  <files>
    scripts/promote_v7.py
  </files>
  <read_first>
    src/acoustic/evaluation/evaluator.py,
    src/acoustic/evaluation/promotion.py,
    src/acoustic/evaluation/uma16_eval.py
  </read_first>
  <behavior>
    `python scripts/promote_v7.py --v7 models/efficientat_mn10_v7.pt --dads-test <path> --uma16-eval data/eval/uma16_real --expected-sha256 <hex>`:
    1. Loads v7 model
    2. Runs Evaluator on the DADS test split → captures dads_acc
    3. Runs Evaluator on the UMA-16 real eval set (via load_uma16_eval_set) → captures real_tpr, real_fpr
    4. Calls promote_v7_if_gates_pass(...) with the measured values
    5. Prints a JSON report (dads_acc, real_tpr, real_fpr, promoted: bool, reason: str|None)
    6. Exits 0 on promote success, 1 on gate fail, 2 on missing files
  </behavior>
  <action>
    Create scripts/promote_v7.py:
    ```python
    """Phase 20 v7 promotion CLI (D-29).

    Runs DADS test eval + UMA-16 real eval, applies the D-29 gate, and copies
    v7 → models/efficientat_mn10.pt iff BOTH thresholds pass.

    Usage:
        python scripts/promote_v7.py \
            --v7 models/efficientat_mn10_v7.pt \
            --dads-test data/dads/test \
            --uma16-eval data/eval/uma16_real \
            --expected-sha256 <hex from 20-05-SUMMARY.md>
    """
    from __future__ import annotations
    import argparse
    import json
    import sys
    from pathlib import Path

    from acoustic.evaluation.uma16_eval import load_uma16_eval_set
    from acoustic.evaluation.promotion import promote_v7_if_gates_pass
    # NOTE: import the project's existing Evaluator the same way Phase 9 tests do.
    from acoustic.evaluation.evaluator import Evaluator  # adjust if class name differs

    def main() -> int:
        ap = argparse.ArgumentParser()
        ap.add_argument("--v7", type=Path, default=Path("models/efficientat_mn10_v7.pt"))
        ap.add_argument("--dads-test", type=Path, required=True)
        ap.add_argument("--uma16-eval", type=Path, required=True)
        ap.add_argument("--expected-sha256", type=str, default=None)
        ap.add_argument("--target", type=Path, default=Path("models/efficientat_mn10.pt"))
        args = ap.parse_args()

        if not args.v7.exists():
            print(json.dumps({"error": f"v7 checkpoint missing: {args.v7}"}))
            return 2
        if not (args.uma16_eval / "labels.json").exists():
            print(json.dumps({"error": f"UMA-16 eval set labels.json missing: {args.uma16_eval}"}))
            return 2

        evaluator = Evaluator(model_path=str(args.v7))

        # D-26: DADS test accuracy
        dads_result = evaluator.evaluate_classifier(args.dads_test)
        dads_acc = float(getattr(dads_result, "accuracy", 0.0))

        # D-27: UMA-16 real-capture TPR/FPR
        uma16_entries = load_uma16_eval_set(args.uma16_eval)
        uma16_result = evaluator.evaluate_uma16(uma16_entries) if hasattr(evaluator, "evaluate_uma16") \
            else evaluator.evaluate_classifier(args.uma16_eval)
        # Extract TPR/FPR from confusion matrix or fields
        tn = float(getattr(uma16_result, "tn", 0))
        fp = float(getattr(uma16_result, "fp", 0))
        fn = float(getattr(uma16_result, "fn", 0))
        tp = float(getattr(uma16_result, "tp", 0))
        real_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        real_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        promoted = promote_v7_if_gates_pass(
            dads_acc=dads_acc,
            real_tpr=real_tpr,
            real_fpr=real_fpr,
            source_path=args.v7,
            target_path=args.target,
            expected_sha256=args.expected_sha256,
        )

        report = {
            "dads_acc": dads_acc,
            "real_tpr": real_tpr,
            "real_fpr": real_fpr,
            "promoted": promoted,
            "thresholds": {"dads_acc_min": 0.95, "real_tpr_min": 0.80, "real_fpr_max": 0.05},
        }
        print(json.dumps(report, indent=2))
        return 0 if promoted else 1

    if __name__ == "__main__":
        sys.exit(main())
    ```

    If the existing Evaluator interface differs from the assumed `evaluate_classifier(path)`,
    adapt the calls in this script to match — read evaluator.py first to confirm the actual
    method name and argument shape.
  </action>
  <verify>
    <automated>python -c "import ast; ast.parse(open('scripts/promote_v7.py').read())" && python scripts/promote_v7.py --help</automated>
  </verify>
  <acceptance_criteria>
    - File scripts/promote_v7.py exists and is valid Python (ast.parse succeeds)
    - `python scripts/promote_v7.py --help` exits 0 and shows --v7, --dads-test, --uma16-eval, --expected-sha256 args
    - `grep -n "promote_v7_if_gates_pass" scripts/promote_v7.py` returns one match
    - `grep -n "dads_acc_min.*0.95" scripts/promote_v7.py` returns one match
    - `grep -n "real_tpr_min.*0.80" scripts/promote_v7.py` returns one match
    - `grep -n "real_fpr_max.*0.05" scripts/promote_v7.py` returns one match
  </acceptance_criteria>
  <done>
    promote_v7.py CLI exists, parses args, calls evaluator + promotion gate, returns the JSON report
    and the correct exit code.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Run promotion gate against trained v7 checkpoint and apply D-29 decision</name>
  <what-built>
    Eval harness (uma16_eval + ROC), promotion gate module, promote_v7.py CLI. Now run them
    against the actual v7 checkpoint produced by Plan 05.
  </what-built>
  <how-to-verify>
    Pre-conditions:
    - models/efficientat_mn10_v7.pt exists (Plan 05 Task 2 completed)
    - data/eval/uma16_real/labels.json exists with ≥20 min labeled segments (Plan 00 Task 3 completed)
    - The DADS test split is downloadable / accessible

    1. Record / read the v7 sha256 from `.planning/phases/20-.../20-05-SUMMARY.md`.

    2. Run the promotion CLI in DRY mode by setting `--target` to a temp path first to validate
       the eval pipeline:
       ```
       python scripts/promote_v7.py \
           --v7 models/efficientat_mn10_v7.pt \
           --dads-test data/dads/test \
           --uma16-eval data/eval/uma16_real \
           --target /tmp/promotion_test_target.pt
       ```
       Inspect the printed JSON. Expected fields: dads_acc, real_tpr, real_fpr, promoted, thresholds.

    3. If the JSON shows `promoted: true`, re-run with the real target path (or just `cp /tmp/promotion_test_target.pt models/efficientat_mn10.pt` since the gate already passed):
       ```
       python scripts/promote_v7.py \
           --v7 models/efficientat_mn10_v7.pt \
           --dads-test data/dads/test \
           --uma16-eval data/eval/uma16_real \
           --expected-sha256 <hex from 20-05-SUMMARY.md>
       ```

    4. Verify the default model file was updated:
       ```
       ls -la models/efficientat_mn10.pt models/efficientat_mn10_v7.pt
       sha256sum models/efficientat_mn10.pt models/efficientat_mn10_v7.pt
       ```
       The two should now have identical sha256.

    5. If the JSON shows `promoted: false`, DO NOT manually copy the file. Read the JSON to see
       which threshold failed (D-26 vs D-27). v6 remains the default model. Type "gate failed"
       below — the orchestrator will route Phase 20 to gap closure (more UMA-16 ambient data,
       more eval data, or revisiting augmentation params).

    Manual verification of D-28 ROC output:
    ```
    python -c "
    import json
    from acoustic.evaluation.evaluator import Evaluator
    from acoustic.evaluation.uma16_eval import load_uma16_eval_set
    e = Evaluator(model_path='models/efficientat_mn10_v7.pt')
    res = e.evaluate_classifier('data/eval/uma16_real')
    print('ROC points:', len(res.roc_curve) if res.roc_curve else 'NONE')
    "
    ```
  </how-to-verify>
  <resume-signal>
    Type "v7 promoted" if the gate passed and models/efficientat_mn10.pt now matches v7.
    Type "gate failed: D-26" or "gate failed: D-27" if the corresponding threshold did not pass —
    the orchestrator will route to gap closure.
  </resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Trained checkpoint → default model file | The promotion step makes a checkpoint the live deployment model. Authenticity matters. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-06-01 | Tampering | v7 checkpoint authenticity | mitigate | promote_v7_if_gates_pass verifies sha256 against the value recorded in 20-05-SUMMARY.md before copying. |
| T-20-06-02 | Elevation of Privilege | Promotion bypass | mitigate | Promotion is hard-gated on three numeric thresholds (0.95/0.80/0.05). No CLI flag bypasses the gate. The only way around it is direct `cp` by the user, which is auditable. |
| T-20-06-03 | Repudiation (silent failure) | Eval harness producing wrong metrics | mitigate | promote_v7.py prints a JSON report with all measured values + thresholds; manual checkpoint requires the user to read the report. |
| T-20-06-04 | Information Disclosure | UMA-16 eval recordings | accept | Recordings are project-internal; no PII. Stored under data/eval/uma16_real (recommend .gitignore). |
</threat_model>

<verification>
- Wave 0 promotion + evaluator tests pass
- promote_v7.py --help works
- Manual run against trained v7 produces JSON report with all four fields and correct exit code
- D-29 enforced: v7 only becomes default model when both thresholds pass
</verification>

<success_criteria>
- `pytest tests/unit/test_promotion_gate.py tests/unit/test_evaluator.py -x -q` exits 0
- `python scripts/promote_v7.py --help` exits 0
- After Task 3: Either models/efficientat_mn10.pt sha256 matches v7 (promoted), OR the user reported a specific gate failure for orchestrator gap-closure routing
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-06-SUMMARY.md`
with the final JSON gate report (dads_acc, real_tpr, real_fpr, promoted) embedded so future
phases / retrospectives can reference the v7 → default decision.
</output>
