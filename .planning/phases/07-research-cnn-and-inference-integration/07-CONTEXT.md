# Phase 7: Research CNN and Inference Integration - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the dormant classifier slot with the research CNN architecture (3-layer Conv2D 32/64/128, BN, MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid), add multi-segment aggregation with configurable weighting, wire the classifier factory in main.py for protocol-based injection at startup, and make state machine thresholds configurable for the new CNN's confidence distribution. This phase does NOT include training (Phase 8), evaluation (Phase 9), or ensemble support (Phase 11).

</domain>

<decisions>
## Implementation Decisions

### Segment Aggregation Strategy
- **D-01:** Aggregation window is 2 seconds (4 overlapping 0.5s segments). Matches the old 2.0s CNN window size, balancing latency vs. stability.
- **D-02:** Segments overlap by 50% (0.25s hop between segments). Standard in audio ML to capture events at segment boundaries.
- **D-03:** Final probability uses weighted combination: `p_agg = w_max * p_max + w_mean * p_mean` with configurable weights (default 0.5/0.5). Research approach balancing peak detection with average confidence.

### Model Loading & Factory Wiring
- **D-04:** When the configured model file doesn't exist at startup, the service boots normally with CNNWorker in dormant mode (classifier=None). Logs a warning. Allows beamforming-only operation until a model is trained in Phase 8.
- **D-05:** Only PyTorch `.pt` format supported. No legacy .h5 or ONNX format support. Clean break — Phase 6 already removed all TF/ONNX dependencies.
- **D-06:** Classifier factory in main.py reads `cnn_model_path` from AcousticSettings, instantiates ResearchCNN, loads state_dict, and injects into CNNWorker. Falls back to None if model file missing.

### State Machine Recalibration
- **D-07:** Keep current thresholds (enter=0.80, exit=0.40, confirm_hits=2) as defaults. They are already configurable via `ACOUSTIC_CNN_ENTER_THRESHOLD`, `ACOUSTIC_CNN_EXIT_THRESHOLD`, `ACOUSTIC_CNN_CONFIRM_HITS` env vars. Phase 9 evaluation will determine optimal values — no guesswork now.
- **D-08:** Aggregated p_agg feeds directly into the existing state machine as the single probability value. No state machine code changes needed — thresholds apply to the aggregated output.

### Aggregator Protocol Design
- **D-09:** New `Aggregator` protocol formalized alongside Classifier/Preprocessor: `aggregate(probabilities: list[float]) -> float`. Injected into CNNWorker as a third protocol dependency. Enables Phase 11 ensemble to swap aggregation strategy cleanly.
- **D-10:** Aggregation weights (w_max, w_mean) configurable via env vars: `ACOUSTIC_CNN_AGG_W_MAX` and `ACOUSTIC_CNN_AGG_W_MEAN` in AcousticSettings. Defaults to 0.5/0.5.

### Claude's Discretion
- PyTorch model class placement (e.g., `classification/model.py` or `classification/research_cnn.py`)
- How CNNWorker manages the segment buffer internally (ring buffer, deque, list)
- Whether to add a model validation step on load (forward pass with dummy tensor)
- How overlapping segments are generated from the audio buffer in pipeline.py
- Default Aggregator implementation class name and location

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Research Reference Implementation
- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` -- Canonical model architecture: `build_model()` (line ~208). 3-layer Conv2D (32/64/128) + BN + MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid. Input shape (128, 64, 1) in TF format = (1, 1, 128, 64) in PyTorch (channels-first). Port this architecture to `nn.Module`.

### Existing Service Code (Phase 7 touches these)
- `src/acoustic/classification/protocols.py` -- Classifier and Preprocessor protocols. Add Aggregator protocol here.
- `src/acoustic/classification/worker.py` -- CNNWorker with dormant classifier slot. Add segment buffering, aggregation, and classifier injection.
- `src/acoustic/classification/config.py` -- MelConfig dataclass. May need aggregation-related constants.
- `src/acoustic/classification/preprocessing.py` -- ResearchPreprocessor already functional. No changes expected.
- `src/acoustic/classification/state_machine.py` -- DetectionStateMachine. No code changes — thresholds already configurable. Receives aggregated p_agg.
- `src/acoustic/config.py` -- AcousticSettings. Add aggregation weight env vars (w_max, w_mean).
- `src/acoustic/main.py` -- Lifespan factory. Add classifier instantiation, aggregator injection, model loading logic.
- `src/acoustic/pipeline.py` -- BeamformingPipeline._process_cnn(). May need changes for multi-segment push strategy.

### Requirements
- `.planning/REQUIREMENTS.md` -- MDL-01, MDL-02, MDL-03, MDL-04 define the acceptance criteria for this phase.

### Prior Phase Context
- `.planning/phases/06-preprocessing-parity-foundation/06-CONTEXT.md` -- Phase 6 decisions on protocol design, preprocessing, ONNX removal.
- `.planning/phases/03-cnn-classification-and-target-tracking/03-CONTEXT.md` -- Phase 3 decisions on state machine, target tracking, event publishing.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ResearchPreprocessor` in `preprocessing.py` -- Already implements Preprocessor protocol, produces (1, 1, 128, 64) tensors. Ready to use.
- `CNNWorker` in `worker.py` -- Protocol-refactored background thread with single-slot queue. Needs segment buffer and aggregator injection added.
- `DetectionStateMachine` in `state_machine.py` -- Three-state hysteresis (NO_DRONE/CANDIDATE/CONFIRMED). Unchanged — receives aggregated probability.
- `TargetTracker` + `EventBroadcaster` -- Event pipeline from confirmed detection to WebSocket. Unchanged.
- `AcousticSettings` in `config.py` -- Pydantic BaseSettings with `ACOUSTIC_` env prefix. Add aggregation weight fields here.
- `MelConfig` in `classification/config.py` -- Frozen dataclass with research parameters. Reference for model input shape.

### Established Patterns
- Protocol-based dependency injection (Classifier, Preprocessor — now Aggregator)
- Config via Pydantic BaseSettings with env var override
- Background daemon thread with single-slot queue for non-blocking inference
- FastAPI lifespan for startup/shutdown of pipeline components
- Energy gating (RMS silence threshold) before CNN inference

### Integration Points
- `main.py` lifespan: `classifier=None` replaced with `ResearchCNN(loaded state_dict)` or `None` if model missing
- `CNNWorker.__init__`: Add `aggregator` parameter alongside `preprocessor` and `classifier`
- `pipeline.py._process_cnn()`: May need adjustment for multi-segment buffering strategy
- `protocols.py`: Add `Aggregator` protocol definition

</code_context>

<specifics>
## Specific Ideas

- Research CNN architecture must exactly match `train_strong_cnn.py` `build_model()` — same layer sizes, activations, and dropout rate
- Input tensor format: PyTorch channels-first (N, 1, 128, 64) vs TF channels-last (N, 128, 64, 1)
- Aggregation window (2s, 4 segments, 50% overlap) mimics the old 2.0s pipeline but with finer-grained segment analysis
- Weighted combo `p_agg = 0.5*p_max + 0.5*p_mean` is the default; tunable via env vars for field experimentation

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-research-cnn-and-inference-integration*
*Context gathered: 2026-04-01*
