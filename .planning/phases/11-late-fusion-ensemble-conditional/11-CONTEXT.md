# Phase 11: Late Fusion Ensemble (Conditional) - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build an EnsembleClassifier that wraps N models (potentially different architectures) via the Classifier protocol with accuracy-weighted soft voting, a model registry for architecture resolution, an ensemble config file for activation, and ensemble-aware evaluation that reports both ensemble and per-model metrics. This phase does NOT include new model architectures (those are trained/added independently), UI for ensemble management (future phase), or automatic ensemble optimization.

</domain>

<decisions>
## Implementation Decisions

### Ensemble Composition
- **D-01:** Ensemble supports different architectures, not just multiple ResearchCNN training runs. Each architecture implements the Classifier protocol independently. Diversity comes from architectural differences (e.g., ResearchCNN, deeper CNN, attention-based models).
- **D-02:** Hard cap of 3 models for live detection mode. EnsembleClassifier rejects >3 models when used in the live pipeline. Matches real-time latency budget from success criteria.
- **D-03:** Model registry mapping pattern: a dict mapping model type string (e.g., "research_cnn", "deep_cnn") to a loader function. Ensemble config lists entries with type + path. Factory resolves each model via the registry. Extensible without code changes.

### Weight Strategy
- **D-04:** Accuracy weights derived from evaluation F1 scores. Run each model through the Phase 9 evaluation harness, use per-model F1 as weight, normalize so weights sum to 1.0.
- **D-05:** Weights are static at startup. Pre-computed from evaluation results and stored in the ensemble config file. Loaded once when service starts. No dynamic recalculation.

### Live vs Offline Modes
- **D-06:** Ensemble activated via config file (JSON). File lists model entries (type, path, weight). If file exists and has >1 model, service starts in ensemble mode. If missing or single model, falls back to single-model mode via existing factory path.
- **D-07:** EnsembleClassifier implements Classifier protocol, so the existing Evaluator.evaluate() works unchanged. For offline evaluation, no model count cap — N models allowed. Ensemble evaluation triggered via existing /api/eval/run endpoint by passing ensemble config.
- **D-08:** When evaluating an ensemble, the response includes ensemble metrics AND per-model individual metrics. Operator sees improvement over single-model at a glance without running separate evaluations.

### Conditionality Gate
- **D-09:** Manual decision after reviewing Phase 9 evaluation results. No hardcoded threshold. Operator judges whether ensemble is worth pursuing based on single-model accuracy, use case needs, and available model variants.

### Claude's Discretion
- EnsembleClassifier class location (e.g., `classification/ensemble.py`)
- Ensemble config file format details (JSON schema)
- How per-model metrics are included in evaluation response (nested in existing EvaluationResult or new response model)
- Model registry implementation details (module-level dict, class, or factory pattern)
- How ensemble config path is specified (env var like `ACOUSTIC_ENSEMBLE_CONFIG` in AcousticSettings)
- Whether soft voting uses weighted average of probabilities or weighted average of logits

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Classification Infrastructure
- `src/acoustic/classification/protocols.py` -- Classifier, Preprocessor, Aggregator protocols. EnsembleClassifier implements Classifier protocol.
- `src/acoustic/classification/research_cnn.py` -- ResearchCNN model and ResearchClassifier wrapper. First model type in the registry.
- `src/acoustic/classification/worker.py` -- CNNWorker accepts `classifier: Classifier | None`. Ensemble injected via same slot.
- `src/acoustic/classification/aggregation.py` -- WeightedAggregator. Segment aggregation happens BEFORE ensemble (each model gets pre-aggregated probability).
- `src/acoustic/classification/config.py` -- MelConfig dataclass. Shared preprocessing constants.

### Evaluation Infrastructure
- `src/acoustic/evaluation/evaluator.py` -- Evaluator class. Runs classifier on labeled test folders. Must work with EnsembleClassifier via Classifier protocol.
- `src/acoustic/evaluation/models.py` -- EvaluationResult, FileResult, DistributionStats Pydantic models. May need extension for per-model metrics.
- `src/acoustic/api/eval_routes.py` -- POST /api/eval/run endpoint. Must support ensemble evaluation.

### Factory and Wiring
- `src/acoustic/main.py` -- Lifespan factory (lines ~297-354). Currently creates ResearchClassifier. Must be extended with ensemble config detection and model registry resolution.
- `src/acoustic/config.py` -- AcousticSettings. Add ensemble config path setting.

### Prior Phase Context
- `.planning/phases/07-research-cnn-and-inference-integration/07-CONTEXT.md` -- Classifier factory pattern (D-06), Aggregator protocol (D-09).
- `.planning/phases/09-evaluation-harness-and-api/09-CONTEXT.md` -- Evaluation harness design, API endpoints, metrics output.

### Requirements
- `.planning/REQUIREMENTS.md` -- ENS-01, ENS-02 define acceptance criteria for this phase.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Classifier` protocol in `protocols.py` -- EnsembleClassifier implements `predict(features) -> float` directly. Zero interface changes needed.
- `ResearchClassifier` in `research_cnn.py` -- First registry entry. Pattern for wrapping models in Classifier protocol.
- `Evaluator` in `evaluator.py` -- Already takes a model path and runs evaluation. Protocol-based, so ensemble works if it implements Classifier.
- `WeightedAggregator` in `aggregation.py` -- Segment aggregation is orthogonal to ensemble. Each model in the ensemble receives the same pre-processed features.

### Established Patterns
- Protocol-based dependency injection (Classifier, Preprocessor, Aggregator)
- Config via Pydantic BaseSettings with `ACOUSTIC_` env prefix
- Factory in main.py lifespan for classifier instantiation
- Dormant mode when model/config missing (log warning, beamforming-only)

### Integration Points
- `main.py` lifespan: Detect ensemble config file -> load models via registry -> create EnsembleClassifier -> inject into CNNWorker
- `CNNWorker.classifier`: Single slot accepts EnsembleClassifier (same Classifier protocol)
- `Evaluator.evaluate()`: Pass EnsembleClassifier as the classifier; extend response for per-model metrics
- `/api/eval/run`: Optionally accept ensemble config to evaluate ensemble vs individual models

</code_context>

<specifics>
## Specific Ideas

- EnsembleClassifier.predict() calls each model's predict() independently, applies F1-derived weights, returns weighted average probability
- Ensemble config file enables clean activation: drop a JSON file with model entries to switch from single to ensemble mode
- Model registry makes adding new architectures a one-line registration, no factory rewrites
- Per-model metrics in evaluation response lets operator see exactly which models contribute and whether ensemble actually improves over best single model

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 11-late-fusion-ensemble-conditional*
*Context gathered: 2026-04-02*
