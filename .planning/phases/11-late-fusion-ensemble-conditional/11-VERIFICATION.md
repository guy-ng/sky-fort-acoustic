---
phase: 11-late-fusion-ensemble-conditional
verified: 2026-04-02T18:30:00Z
status: human_needed
score: 8/8 must-haves verified
re_verification: false
human_verification:
  - test: "Start the service with an ensemble config JSON pointing to two trained ResearchCNN .pt files and verify the startup log says 'Loaded ensemble with 2 models'"
    expected: "Service starts without errors, ensemble mode logged, predictions flow from the weighted combination of both models"
    why_human: "Requires real trained .pt model files and a running service; cannot be exercised without hardware or a trained checkpoint"
  - test: "Run evaluate_ensemble via the API on a labeled test dataset with two real models and confirm per_model_results has 2 entries AND ensemble accuracy exceeds the best single-model score"
    expected: "Response body includes per_model_results array with two entries, ensemble accuracy >= best single-model F1"
    why_human: "Success Criterion 3 (measurable improvement over best single-model baseline) cannot be verified without trained models and real labeled data — unit tests use mock classifiers that cannot demonstrate accuracy improvement"
---

# Phase 11: Late Fusion Ensemble (Conditional) Verification Report

**Phase Goal:** Multiple classifiers combine via accuracy-weighted soft voting to improve detection accuracy beyond what a single model achieves
**Verified:** 2026-04-02T18:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EnsembleClassifier combines N model predictions via weighted soft voting into a single probability | VERIFIED | `EnsembleClassifier.predict` in `ensemble.py` lines 109-114; `test_weighted_soft_voting` passes with correct arithmetic |
| 2 | Weights are F1-derived and normalized to sum to 1.0 at construction | VERIFIED | Lines 106-107 in `ensemble.py`: `total = sum(weights); self._weights = [w / total for w in weights]`; `test_weight_normalization` confirms sum == 1.0 |
| 3 | Live mode rejects >3 models; offline mode allows unlimited | VERIFIED | Lines 100-104 in `ensemble.py`; `test_live_mode_cap` (4 models → ValueError "max 3 models") and `test_offline_no_cap` (5 models → succeeds) pass |
| 4 | Model registry resolves type string to Classifier loader function | VERIFIED | `_REGISTRY` dict, `register_model`, `load_model` in `ensemble.py` lines 53-72; `test_model_registry` and `test_registry_unknown_type` pass; `test_research_cnn_registered` confirms "research_cnn" pre-registered |
| 5 | Ensemble config JSON file parsed into validated Pydantic model | VERIFIED | `EnsembleConfig.from_file` in `ensemble.py` lines 39-44; `test_config_parsing` and `test_config_invalid_json` pass |
| 6 | AcousticSettings exposes ensemble_config_path as optional env var | VERIFIED | `src/acoustic/config.py` line 53: `ensemble_config_path: str | None = None` |
| 7 | Evaluator accepts a Classifier instance; evaluate_ensemble returns per-model metrics alongside ensemble metrics | VERIFIED | `evaluate_classifier` (line 68) and `evaluate_ensemble` (line 115) in `evaluator.py`; `test_evaluate_classifier_with_mock` and `test_evaluate_ensemble_per_model_results` pass |
| 8 | Eval API endpoint supports ensemble evaluation and service lifespan wires ensemble factory | VERIFIED | `eval_routes.py` checks `body.ensemble_config_path` and calls `EnsembleConfig.from_file`; `main.py` lines 308-350 load ensemble config before single-model fallback; integration tests `test_eval_ensemble_missing_config` and `test_eval_ensemble_endpoint_accepts_param` pass |

**Score:** 8/8 truths verified (automated checks)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/acoustic/classification/ensemble.py` | EnsembleClassifier, ModelRegistry, EnsembleConfig, ModelEntry | VERIFIED | 144 lines; exports all 4 symbols; "research_cnn" registered at module level |
| `src/acoustic/config.py` | ensemble_config_path setting | VERIFIED | Line 53: `ensemble_config_path: str | None = None` |
| `tests/unit/test_ensemble.py` | 12 unit tests for all ensemble behaviors | VERIFIED | 157 lines; 12 test functions covering all specified behaviors |
| `src/acoustic/evaluation/evaluator.py` | evaluate_classifier, evaluate_ensemble | VERIFIED | 357 lines; both methods implemented with single-pass ensemble evaluation |
| `src/acoustic/evaluation/models.py` | PerModelResult dataclass, per_model_results on EvaluationResult | VERIFIED | Lines 32-41 (PerModelResult); line 65 (per_model_results field) |
| `src/acoustic/api/models.py` | PerModelResultResponse, ensemble_config_path on EvalRunRequest, per_model_results on EvalResultResponse | VERIFIED | All three present; from_evaluation converts per_model_results when non-empty |
| `src/acoustic/api/eval_routes.py` | Ensemble evaluation endpoint path | VERIFIED | Lines 53-105; validates config file, model files, creates EnsembleClassifier with live_mode=False |
| `src/acoustic/main.py` | Ensemble factory in lifespan with live_mode=True and single-model fallback | VERIFIED | Lines 308-370; ensemble factory checked first, single-model fallback conditional on classifier is None |
| `tests/unit/test_evaluator.py` | test_evaluate_classifier_with_mock, test_evaluate_ensemble_per_model_results | VERIFIED | Both test classes present and passing (TestEvaluateClassifierWithMock, TestEvaluateEnsemblePerModelResults) |
| `tests/integration/test_eval_api.py` | test_eval_ensemble_missing_config, test_eval_ensemble_endpoint_accepts_param | VERIFIED | TestEvalEnsemble class with both tests; both pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ensemble.py` | `protocols.py` | EnsembleClassifier implements Classifier protocol | VERIFIED | `predict(self, features: torch.Tensor) -> float` at line 109; `isinstance(ensemble, Classifier)` passes in test_protocol_compliance |
| `ensemble.py` | `research_cnn.py` | register_model registers research_cnn loader | VERIFIED | Line 143: `register_model("research_cnn", _load_research_cnn)`; test_research_cnn_registered passes |
| `evaluator.py` | `ensemble.py` | evaluate_classifier accepts EnsembleClassifier as Classifier | VERIFIED | Signature `evaluate_classifier(self, classifier: Classifier, ...)` at line 68; test uses EnsembleClassifier with mock sub-classifiers |
| `main.py` | `ensemble.py` | Factory loads ensemble config and creates EnsembleClassifier | VERIFIED | Lines 315-317 import EnsembleClassifier, EnsembleConfig; line 321 calls EnsembleConfig.from_file; line 329 creates EnsembleClassifier |
| `eval_routes.py` | `ensemble.py` | Eval endpoint loads ensemble config for evaluation | VERIFIED | Lines 69-73 import and use EnsembleConfig.from_file, EnsembleClassifier |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `EnsembleClassifier.predict` | weighted sum of `clf.predict(features)` | N classifier instances in `_classifiers` list | Yes — classifiers loaded from registry which wraps real model checkpoints | FLOWING |
| `Evaluator.evaluate_ensemble` | `result.per_model_results` | `_compute_confusion` / `_compute_metrics` on per-model file results in single-pass loop | Yes — real WAV files processed through individual classifiers | FLOWING |
| `EvalResultResponse.from_evaluation` | `per_model_results` field | `result.per_model_results` (non-empty for ensemble, empty list for single-model) | Yes — conditionally populated from domain result | FLOWING |
| `main.py` lifespan `classifier` | `EnsembleClassifier` instance | `EnsembleConfig.from_file(settings.ensemble_config_path)` then `load_model` per entry | Yes — loads from real .pt files at startup (falls back to None if files absent) | FLOWING (startup-time; requires real model files at runtime) |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| EnsembleClassifier weighted voting arithmetic | `pytest tests/unit/test_ensemble.py::test_weighted_soft_voting -q` | PASS (0.7125 exact) | PASS |
| Weight normalization | `pytest tests/unit/test_ensemble.py::test_weight_normalization -q` | PASS | PASS |
| Live mode cap enforced | `pytest tests/unit/test_ensemble.py::test_live_mode_cap -q` | PASS | PASS |
| evaluate_ensemble per-model count | `pytest tests/unit/test_evaluator.py::TestEvaluateEnsemblePerModelResults -q` | PASS (2 entries for 2 mock models) | PASS |
| Ensemble API missing config 404 | `pytest tests/integration/test_eval_api.py::TestEvalEnsemble::test_eval_ensemble_missing_config -q` | PASS | PASS |
| Full test suite (excluding unrelated pre-existing failures) | `pytest tests/unit/test_ensemble.py tests/unit/test_evaluator.py tests/integration/test_eval_api.py` | 34/34 passed | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ENS-01 | 11-01-PLAN, 11-02-PLAN | NOT FOUND in REQUIREMENTS.md | ORPHANED | Both plans declare ENS-01 and ENS-02 in their `requirements:` fields, but REQUIREMENTS.md has no ENS section. The intent of ENS-01 (EnsembleClassifier with weighted soft voting) is implemented; the formal requirement definition is absent from the tracking document. |
| ENS-02 | 11-01-PLAN, 11-02-PLAN | NOT FOUND in REQUIREMENTS.md | ORPHANED | Same as ENS-01 — implemented but not formally tracked in REQUIREMENTS.md. ENS-02 (ensemble evaluation showing improvement over single model) is partially verifiable: the infrastructure exists and evaluation produces per-model metrics, but the "measurable improvement" assertion requires real trained models. |

**Note:** ENS-01 and ENS-02 are ORPHANED requirements — they are referenced by both plans but have no entry in `.planning/REQUIREMENTS.md`. The implementation clearly covers the semantic intent of both IDs. The requirements file should be updated to define these IDs under an "Ensemble" section.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/acoustic/evaluation/evaluator.py` | 115-119 | `evaluate_ensemble` parameter types use `object` and `list` instead of typed imports | Info | Type hints weakened (`ensemble: object`, `model_entries: list`) to avoid circular imports — types are imported lazily inside the method body at line 134. Functionally correct but reduces static analysis coverage. |

No placeholder, stub, or TODO patterns found in any phase 11 files.

---

### Human Verification Required

The automated checks confirm all structural and behavioral requirements are met by the implementation. Two items require human verification because they depend on trained model files and real labeled audio data.

#### 1. Ensemble Live Pipeline with Real Models

**Test:** Build an ensemble config JSON with two trained ResearchCNN checkpoint paths. Set `ACOUSTIC_ENSEMBLE_CONFIG_PATH` to that file and start the service. Check startup logs.
**Expected:** Log line "Loaded ensemble with 2 models from ..." appears; health endpoint shows pipeline_running=true; detection events from WebSocket reflect weighted ensemble predictions.
**Why human:** Requires real .pt checkpoint files and a running service instance. Cannot verify without trained models or audio hardware.

#### 2. Ensemble Accuracy Improvement Over Single Baseline (Success Criterion 3)

**Test:** Run `POST /api/eval/run` with `ensemble_config_path` pointing to an ensemble of 2+ trained models and a labeled test dataset. Compare ensemble F1 to best single-model F1 from a prior single-model evaluation.
**Expected:** Ensemble F1 >= best single-model F1 (measurable improvement per phase goal). `per_model_results` in response shows individual model metrics for comparison.
**Why human:** Success Criterion 3 requires real trained models with different error profiles. Mock classifiers in unit tests use fixed probabilities that demonstrate the mechanism but cannot prove accuracy improvement over a baseline.

---

### Gaps Summary

No blocking gaps. All 8 observable truths verified programmatically. All artifacts exist, are substantive, and are correctly wired. 34 tests pass (12 unit ensemble, 16 unit evaluator, 6 integration eval API).

**One documentation gap:** ENS-01 and ENS-02 requirement IDs are referenced in plan frontmatter but are not defined in `.planning/REQUIREMENTS.md`. This is a tracking artifact gap, not an implementation gap. The implementation satisfies the semantic intent of both requirements.

**Two items require human verification** before the phase goal can be fully declared achieved: (1) live ensemble pipeline with real model files, and (2) demonstrable accuracy improvement over the best single-model baseline (Success Criterion 3).

---

_Verified: 2026-04-02T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
