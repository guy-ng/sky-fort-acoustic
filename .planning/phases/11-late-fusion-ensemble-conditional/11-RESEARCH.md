# Phase 11: Late Fusion Ensemble (Conditional) - Research

**Researched:** 2026-04-02
**Domain:** Late fusion ensemble classification, accuracy-weighted soft voting, model registry
**Confidence:** HIGH

## Summary

This phase adds an `EnsembleClassifier` that wraps N models (each implementing the existing `Classifier` protocol) and combines their predictions via accuracy-weighted soft voting. The existing codebase provides an exceptionally clean integration surface: `CNNWorker` accepts any `Classifier`, the `Evaluator` runs inference through models directly, and `AcousticSettings` uses pydantic-settings with `ACOUSTIC_` prefix for configuration.

The implementation is straightforward Python -- no new libraries are needed. The ensemble pattern is a weighted average of probabilities from independently-loaded models, gated by a JSON config file. The primary complexity is in the factory wiring (main.py lifespan) and extending the evaluation response to include per-model metrics.

**Primary recommendation:** Implement EnsembleClassifier as a simple class in `classification/ensemble.py` that implements the Classifier protocol, with a model registry dict and JSON config file for activation. Extend EvaluationResult/EvalResultResponse to carry per-model metrics alongside ensemble metrics.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Ensemble supports different architectures, not just multiple ResearchCNN training runs. Each architecture implements the Classifier protocol independently.
- **D-02:** Hard cap of 3 models for live detection mode. EnsembleClassifier rejects >3 models when used in the live pipeline.
- **D-03:** Model registry mapping pattern: a dict mapping model type string to a loader function. Ensemble config lists entries with type + path. Factory resolves each model via the registry.
- **D-04:** Accuracy weights derived from evaluation F1 scores. Per-model F1 as weight, normalized so weights sum to 1.0.
- **D-05:** Weights are static at startup. Pre-computed from evaluation results and stored in the ensemble config file. Loaded once.
- **D-06:** Ensemble activated via config file (JSON). If file exists and has >1 model, service starts in ensemble mode.
- **D-07:** EnsembleClassifier implements Classifier protocol. Existing Evaluator works unchanged. For offline evaluation, no model count cap.
- **D-08:** Evaluation response includes ensemble metrics AND per-model individual metrics.
- **D-09:** Manual decision to activate ensemble -- no hardcoded threshold.

### Claude's Discretion
- EnsembleClassifier class location (recommend: `classification/ensemble.py`)
- Ensemble config file format details (JSON schema)
- How per-model metrics are included in evaluation response
- Model registry implementation details
- How ensemble config path is specified (env var)
- Whether soft voting uses weighted average of probabilities or logits

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENS-01 | EnsembleClassifier wraps N models via Classifier protocol with accuracy-weighted soft voting and normalized weights | Classifier protocol exists in `protocols.py`, `predict(features) -> float` signature. Weighted average of probabilities is trivial. Model registry + JSON config for loading. |
| ENS-02 | Ensemble inference runs within real-time latency budget (max 3 models live, N models offline); ensemble evaluation shows measurable improvement over best single-model | Hard cap enforced in constructor/factory. Sequential inference of 3 models is ~3x single-model latency (~15-45ms total, well within 150ms chunk budget). Evaluator extended to report per-model + ensemble metrics. |

</phase_requirements>

## Standard Stack

### Core
No new libraries required. This phase uses only existing project dependencies.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.11 | Model loading and inference | Already in project, all models are PyTorch |
| pydantic | (existing) | Ensemble config validation, response models | Already used for AcousticSettings and API models |
| pydantic-settings | (existing) | `ACOUSTIC_ENSEMBLE_CONFIG` env var | AcousticSettings pattern established |

### Supporting
No additional supporting libraries needed.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual weighted average | scikit-learn VotingClassifier | Overkill for binary classification with 2-3 models. Adds dependency for 5 lines of code. |
| JSON config file | YAML/TOML config | JSON is simpler, no extra dependency, Python stdlib `json` module. |
| Probability averaging | Logit averaging | Probability averaging is simpler, more interpretable, and standard for late fusion with soft voting. Logit averaging can be numerically unstable with sigmoid outputs near 0/1. **Recommend probability averaging.** |

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/classification/
    ensemble.py          # EnsembleClassifier + ModelRegistry + EnsembleConfig
    protocols.py         # Classifier protocol (unchanged)
    research_cnn.py      # First registered model type (unchanged)
    ...
src/acoustic/evaluation/
    evaluator.py         # Extended: accept Classifier protocol, return per-model metrics
    models.py            # Extended: EnsembleEvaluationResult or per-model fields
src/acoustic/api/
    models.py            # Extended: EvalResultResponse with per_model_results
    eval_routes.py       # Extended: ensemble config parameter support
src/acoustic/
    config.py            # AcousticSettings + ensemble_config_path
    main.py              # Lifespan factory: ensemble detection + model registry resolution
```

### Pattern 1: EnsembleClassifier implementing Classifier Protocol
**What:** A class that holds N classifiers and their weights, calls each model's `predict()`, returns weighted average probability.
**When to use:** Always -- this is the core of the phase.
**Example:**
```python
from acoustic.classification.protocols import Classifier

class EnsembleClassifier:
    """Late fusion ensemble via accuracy-weighted soft voting."""

    def __init__(
        self,
        classifiers: list[Classifier],
        weights: list[float],
        *,
        max_live_models: int = 3,
        live_mode: bool = True,
    ) -> None:
        if live_mode and len(classifiers) > max_live_models:
            raise ValueError(
                f"Live mode allows max {max_live_models} models, got {len(classifiers)}"
            )
        if len(classifiers) != len(weights):
            raise ValueError("classifiers and weights must have same length")

        self._classifiers = classifiers
        # Normalize weights to sum to 1.0
        total = sum(weights)
        self._weights = [w / total for w in weights] if total > 0 else weights

    def predict(self, features: torch.Tensor) -> float:
        """Weighted average of all model predictions."""
        weighted_sum = 0.0
        for classifier, weight in zip(self._classifiers, self._weights):
            prob = classifier.predict(features)
            weighted_sum += weight * prob
        return weighted_sum
```

### Pattern 2: Model Registry
**What:** A module-level dict mapping type strings to loader functions.
**When to use:** When resolving ensemble config entries to Classifier instances.
**Example:**
```python
import torch
from typing import Callable
from acoustic.classification.protocols import Classifier

# Type alias for loader: takes a file path, returns a Classifier
ModelLoader = Callable[[str], Classifier]

# Registry: maps type string -> loader function
_REGISTRY: dict[str, ModelLoader] = {}

def register_model(type_name: str, loader: ModelLoader) -> None:
    """Register a model type with its loader function."""
    _REGISTRY[type_name] = loader

def load_model(type_name: str, path: str) -> Classifier:
    """Load a model by type name and file path."""
    if type_name not in _REGISTRY:
        raise ValueError(f"Unknown model type: {type_name}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[type_name](path)

# Register ResearchCNN at import time
def _load_research_cnn(path: str) -> Classifier:
    from acoustic.classification.research_cnn import ResearchClassifier, ResearchCNN
    model = ResearchCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return ResearchClassifier(model)

register_model("research_cnn", _load_research_cnn)
```

### Pattern 3: Ensemble Config File (JSON)
**What:** A JSON file that activates ensemble mode and defines model entries.
**When to use:** Drop a config file to switch from single-model to ensemble mode.
**Example:**
```json
{
  "models": [
    {
      "type": "research_cnn",
      "path": "models/uav_melspec_cnn_v1.pt",
      "weight": 0.85
    },
    {
      "type": "research_cnn",
      "path": "models/uav_melspec_cnn_v2.pt",
      "weight": 0.92
    }
  ]
}
```
The `weight` field stores the pre-computed F1 score from evaluation. EnsembleClassifier normalizes these at construction time.

### Pattern 4: Factory Extension in main.py Lifespan
**What:** Check for ensemble config file, if present load models via registry, create EnsembleClassifier, inject into CNNWorker. Falls back to existing single-model path if no ensemble config.
**When to use:** Service startup.
**Key logic:**
```python
# In main.py lifespan, after existing classifier factory code:
ensemble_config_path = settings.ensemble_config_path
if ensemble_config_path and os.path.isfile(ensemble_config_path):
    # Load ensemble
    config = EnsembleConfig.from_file(ensemble_config_path)
    if len(config.models) > 1:
        classifiers = []
        weights = []
        for entry in config.models:
            clf = load_model(entry.type, entry.path)
            classifiers.append(clf)
            weights.append(entry.weight)
        classifier = EnsembleClassifier(
            classifiers, weights, live_mode=True
        )
        logger.info("Loaded ensemble with %d models", len(classifiers))
    # else: single model in config, use it directly
```

### Anti-Patterns to Avoid
- **Running models in parallel threads:** For 2-3 lightweight CNNs, sequential inference is simpler and avoids thread synchronization overhead. The total latency (~15-45ms for 3 models) is well within the 150ms chunk budget. Parallel execution adds complexity with minimal gain.
- **Dynamic weight recalculation:** Per D-05, weights are static. Do not add runtime F1 tracking or adaptive weighting. It adds complexity and makes behavior unpredictable.
- **Modifying Classifier protocol:** The protocol is `predict(features: torch.Tensor) -> float`. EnsembleClassifier must implement this exact signature. Do not add ensemble-specific methods to the protocol.
- **Tight coupling to ResearchCNN in Evaluator:** The current `Evaluator.evaluate()` method hard-codes `ResearchCNN()` instantiation (line 56-59). This must be refactored to accept a `Classifier` directly or use the model registry, so ensemble evaluation works via protocol dispatch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON config parsing | Custom parser | Pydantic BaseModel with `model_validate_json()` | Validation, type coercion, error messages for free |
| Weight normalization | Complex normalization logic | Simple `w / sum(weights)` | Trivially correct for positive F1 weights |
| Model loading dispatch | if/elif chain on model type | Registry dict with callable loaders | Extensible, clean, one-line registration per type |

## Common Pitfalls

### Pitfall 1: Evaluator Hard-Codes ResearchCNN
**What goes wrong:** The existing `Evaluator.evaluate()` method directly instantiates `ResearchCNN()` on line 56. It cannot evaluate an ensemble or any non-ResearchCNN model.
**Why it happens:** The evaluator was built for Phase 9 with only one model type.
**How to avoid:** Refactor `Evaluator.evaluate()` to accept a `Classifier` instance (or use the model registry to load by type+path). The ensemble evaluation path creates an EnsembleClassifier and passes it in.
**Warning signs:** Tests that only work with ResearchCNN model paths.

### Pitfall 2: Weights Not Normalized
**What goes wrong:** If F1 weights don't sum to 1.0, ensemble probability is not a valid probability (can exceed 1.0 or be too low).
**Why it happens:** Storing raw F1 scores (e.g., 0.85, 0.92) and using them directly.
**How to avoid:** Always normalize in EnsembleClassifier constructor: `w_i / sum(w)`.
**Warning signs:** Ensemble probabilities > 1.0 or detection thresholds behaving unexpectedly.

### Pitfall 3: Feature Tensor Sharing Between Models
**What goes wrong:** If models share the same feature tensor object and one modifies it in-place, subsequent models get corrupted input.
**Why it happens:** PyTorch tensors are mutable; a model with in-place operations would corrupt shared input.
**How to avoid:** The current `predict()` calls use `torch.no_grad()` and standard layers that don't modify input. This is safe for the current ResearchCNN architecture. If future models use in-place operations, clone the tensor before each predict call. Add a defensive `.clone()` only if needed.
**Warning signs:** Different probability values when running models individually vs in ensemble.

### Pitfall 4: Ensemble Config Path Not Set
**What goes wrong:** Service always runs in single-model mode because the env var is never set.
**Why it happens:** New setting not documented or not included in Docker environment.
**How to avoid:** Default to `None` (no ensemble). Log clearly whether ensemble mode is active or inactive at startup.
**Warning signs:** Logs always show "single model mode" even when config file exists.

### Pitfall 5: Evaluation Response Breaking Change
**What goes wrong:** Adding per-model metrics to `EvalResultResponse` breaks existing API consumers (frontend).
**Why it happens:** New required fields added to the response model.
**How to avoid:** Make per-model fields optional with defaults. `per_model_results: list[...] | None = None`. Only populated when evaluating an ensemble.
**Warning signs:** Frontend evaluation display crashes after backend update.

## Code Examples

### Ensemble Config Pydantic Model
```python
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel

class ModelEntry(BaseModel):
    """Single model entry in ensemble configuration."""
    type: str           # Registry key, e.g. "research_cnn"
    path: str           # Path to .pt checkpoint
    weight: float       # F1 score from evaluation (normalized at runtime)

class EnsembleConfig(BaseModel):
    """Ensemble configuration loaded from JSON file."""
    models: list[ModelEntry]

    @classmethod
    def from_file(cls, path: str | Path) -> EnsembleConfig:
        """Load ensemble config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
```

### Extended Evaluation for Ensemble + Per-Model Metrics
```python
@dataclass
class PerModelResult:
    """Evaluation metrics for a single model within an ensemble."""
    model_type: str
    model_path: str
    weight: float          # Normalized weight used in ensemble
    accuracy: float
    precision: float
    recall: float
    f1: float

# Add to EvaluationResult:
# per_model_results: list[PerModelResult] = field(default_factory=list)
```

### AcousticSettings Extension
```python
# Add to AcousticSettings class:
ensemble_config_path: str | None = None
# Env var: ACOUSTIC_ENSEMBLE_CONFIG_PATH
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x with pytest-asyncio |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENS-01a | EnsembleClassifier combines N models via weighted soft voting | unit | `pytest tests/unit/test_ensemble.py::test_weighted_soft_voting -x` | Wave 0 |
| ENS-01b | Weights are normalized to sum to 1.0 | unit | `pytest tests/unit/test_ensemble.py::test_weight_normalization -x` | Wave 0 |
| ENS-01c | Model registry resolves type string to loader | unit | `pytest tests/unit/test_ensemble.py::test_model_registry -x` | Wave 0 |
| ENS-01d | Ensemble config parsed from JSON | unit | `pytest tests/unit/test_ensemble.py::test_config_parsing -x` | Wave 0 |
| ENS-02a | Live mode rejects >3 models | unit | `pytest tests/unit/test_ensemble.py::test_live_mode_cap -x` | Wave 0 |
| ENS-02b | Offline mode allows >3 models | unit | `pytest tests/unit/test_ensemble.py::test_offline_no_cap -x` | Wave 0 |
| ENS-02c | EnsembleClassifier satisfies Classifier protocol | unit | `pytest tests/unit/test_ensemble.py::test_protocol_compliance -x` | Wave 0 |
| ENS-02d | Evaluator works with EnsembleClassifier | unit | `pytest tests/unit/test_evaluator.py -x` (extend existing) | Partial |
| ENS-02e | Eval API returns per-model metrics for ensemble | integration | `pytest tests/integration/test_eval_api.py -x` (extend existing) | Partial |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_ensemble.py -x -q`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_ensemble.py` -- covers ENS-01a through ENS-02c (new file)
- [ ] Extend `tests/unit/test_evaluator.py` -- cover ensemble classifier evaluation
- [ ] Extend `tests/integration/test_eval_api.py` -- cover ensemble eval endpoint

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single model inference | Ensemble of diverse architectures | Standard ML practice | 2-5% accuracy improvement typical for diverse model ensembles |
| Equal weighting | Performance-weighted voting | Standard practice | Better models contribute more, prevents weak model dilution |
| Hard voting (majority) | Soft voting (probability averaging) | Standard for binary classification | Preserves confidence information, better calibrated outputs |

**Domain notes:**
- Late fusion (combining predictions) is simpler and more robust than early fusion (combining features). For binary classification with 2-3 models, late fusion is the standard approach.
- Accuracy-weighted soft voting is the most common ensemble method. More complex approaches (stacking, boosting) are overkill for 2-3 models.
- Ensemble benefit requires model diversity. Multiple runs of the same architecture with different random seeds typically give only marginal improvement. Different architectures (per D-01) provide real diversity.

## Open Questions

1. **How should Evaluator be refactored?**
   - What we know: Current `Evaluator.evaluate()` hard-codes `ResearchCNN()` instantiation and takes `model_path: str`. For ensemble, it needs to accept a `Classifier` instance.
   - What's unclear: Whether to add a separate method (`evaluate_classifier(classifier, data_dir)`) or refactor the existing method signature.
   - Recommendation: Add an overloaded method or refactor to accept `Classifier | str`. Keep backward compatibility for the existing single-model path by using the registry to load from path when a string is passed.

2. **Per-model evaluation: run once or N+1 times?**
   - What we know: D-08 requires both ensemble and per-model metrics in one response.
   - What's unclear: Whether to run inference N+1 times (once per model + once as ensemble) or run once and collect individual + ensemble results in a single pass.
   - Recommendation: Single pass -- during ensemble evaluation, collect each model's prediction per file, compute per-model metrics from those predictions, then compute ensemble metrics from the weighted combination. One inference pass per model, zero extra passes.

## Sources

### Primary (HIGH confidence)
- `src/acoustic/classification/protocols.py` -- Classifier protocol definition (`predict(features: torch.Tensor) -> float`)
- `src/acoustic/classification/research_cnn.py` -- ResearchClassifier wrapper pattern
- `src/acoustic/classification/worker.py` -- CNNWorker accepts `Classifier | None`, single slot injection
- `src/acoustic/evaluation/evaluator.py` -- Current evaluator with hard-coded ResearchCNN
- `src/acoustic/evaluation/models.py` -- EvaluationResult dataclass structure
- `src/acoustic/api/eval_routes.py` -- POST /api/eval/run endpoint
- `src/acoustic/api/models.py` -- EvalRunRequest, EvalResultResponse
- `src/acoustic/config.py` -- AcousticSettings with ACOUSTIC_ prefix
- `src/acoustic/main.py` lines 297-354 -- Classifier factory in lifespan

### Secondary (MEDIUM confidence)
- ML ensemble best practices -- standard textbook late fusion / soft voting approach

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, pure Python implementation using existing project patterns
- Architecture: HIGH -- Classifier protocol provides clean integration surface, patterns are straightforward
- Pitfalls: HIGH -- identified from direct code reading (Evaluator hard-coding, weight normalization, API compatibility)

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable -- no external dependencies changing)
