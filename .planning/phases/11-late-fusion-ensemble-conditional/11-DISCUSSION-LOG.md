# Phase 11: Late Fusion Ensemble (Conditional) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 11-late-fusion-ensemble-conditional
**Areas discussed:** Ensemble composition, Weight strategy, Live vs offline modes, Conditionality gate

---

## Ensemble Composition

### Model Diversity

| Option | Description | Selected |
|--------|-------------|----------|
| Same arch, different runs | Multiple ResearchCNN models trained on different data splits or augmentation configs | |
| Different architectures | Mix ResearchCNN with other architectures (e.g., deeper CNN, attention-based) | ✓ |
| Same arch, different configs | ResearchCNN variants with different hyperparams (dropout, layer sizes) | |

**User's choice:** Different architectures
**Notes:** Supports mixing fundamentally different model types for maximum diversity.

### Model Count Cap

| Option | Description | Selected |
|--------|-------------|----------|
| Hard cap at 3 | EnsembleClassifier rejects >3 models in live mode | ✓ |
| Soft warning only | Log warning if >3 but allow it | |
| Configurable cap | Env var with default of 3 | |

**User's choice:** Hard cap at 3
**Notes:** Matches success criteria latency budget.

### Model Registration

| Option | Description | Selected |
|--------|-------------|----------|
| Model registry mapping | Dict mapping model type string to loader function | ✓ |
| Metadata in checkpoint | Save model type in .pt checkpoint file | |
| Separate config per model | YAML/JSON config file per model | |

**User's choice:** Model registry mapping
**Notes:** Extensible without code changes.

---

## Weight Strategy

### Weight Derivation

| Option | Description | Selected |
|--------|-------------|----------|
| From evaluation F1 scores | Use F1 from eval harness, normalized to sum to 1.0 | ✓ |
| From evaluation accuracy | Use raw accuracy as weight | |
| Manual weights | Operator sets weights via config | |

**User's choice:** From evaluation F1 scores
**Notes:** F1 more meaningful for potentially imbalanced drone/no-drone splits.

### Weight Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Static at startup | Pre-computed, stored in config, loaded once | ✓ |
| Computed on first eval | Ensemble triggers eval at startup to derive weights | |
| Updateable via API | Static at startup but REST endpoint can recalculate | |

**User's choice:** Static at startup
**Notes:** Simple and predictable.

---

## Live vs Offline Modes

### Ensemble Activation

| Option | Description | Selected |
|--------|-------------|----------|
| Ensemble config file | JSON file listing model entries (type, path, weight) | ✓ |
| Env var with model list | Comma-separated model paths with weights | |
| Auto-detect from directory | Scan model directory for all .pt files | |

**User's choice:** Ensemble config file (JSON)
**Notes:** Clean separation, readable for multiple models.

### Offline Evaluation

| Option | Description | Selected |
|--------|-------------|----------|
| Pass ensemble to evaluator | EnsembleClassifier implements Classifier protocol, works with existing Evaluator | ✓ |
| Separate ensemble evaluation | Dedicated endpoint evaluating each model + ensemble | |
| Ensemble is live-only | No ensemble evaluation support | |

**User's choice:** Yes -- pass ensemble to evaluator
**Notes:** No model count cap for offline. Protocol makes this seamless.

---

## Conditionality Gate

### Build Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| Manual decision after eval | Operator reviews Phase 9 results and decides | ✓ |
| F1 threshold gate | Auto-trigger if best single-model F1 < 0.90 | |
| Always build it | Build regardless of single-model accuracy | |

**User's choice:** Manual decision after eval
**Notes:** No hardcoded threshold. Judgment call based on accuracy, needs, and available models.

### Comparison Reporting

| Option | Description | Selected |
|--------|-------------|----------|
| Eval endpoint returns both | Ensemble metrics AND per-model individual metrics in response | ✓ |
| Separate eval runs | Operator runs eval per model manually | |
| Comparison report endpoint | Dedicated /api/eval/compare endpoint | |

**User's choice:** Eval endpoint returns both
**Notes:** Operator sees improvement at a glance.

---

## Claude's Discretion

- EnsembleClassifier class location and internal structure
- Ensemble config JSON schema details
- Model registry implementation pattern
- How per-model metrics are nested in evaluation response
- Soft voting implementation (weighted probability average)
- Ensemble config path env var naming

## Deferred Ideas

None -- discussion stayed within phase scope
