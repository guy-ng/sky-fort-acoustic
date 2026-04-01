# Phase 6: Preprocessing Parity Foundation - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish shared preprocessing configuration with research-validated parameters, implement Classifier/Preprocessor protocols for clean model swaps, build the research mel-spectrogram preprocessor using torchaudio, and validate numerical parity against saved TF/librosa reference tensors. This phase does NOT implement the ResearchCNN model (Phase 7) or training pipeline (Phase 8).

</domain>

<decisions>
## Implementation Decisions

### Old Pipeline Removal
- **D-01:** The existing EfficientNet-B0 preprocessing pipeline is non-functional and will be removed entirely. No coexistence strategy needed — replace `preprocess_for_cnn()` and all EfficientNet-specific code (224x224 resize, 3-channel repeat, z-score normalization).
- **D-02:** Remove `OnnxDroneClassifier` and all ONNX runtime references. The ONNX model is dead. Delete `onnxruntime` from dependencies. The pipeline will have no working classifier until Phase 7 adds ResearchCNN.
- **D-03:** `CNNWorker` should be updated to use the new protocols but will effectively be dormant (no classifier to inject) until Phase 7. Keep the worker structure for Phase 7 to plug into.

### Shared MelConfig
- **D-04:** A single `MelConfig` dataclass centralizes all preprocessing constants: SR=16000, N_FFT=1024, HOP_LENGTH=256, N_MELS=64, MAX_FRAMES=128, SEGMENT_SECONDS=0.5, normalization=(S_db+80)/80 clipped to [0,1]. Eliminates all scattered magic numbers.

### Protocol Design
- **D-05:** Minimal protocols only. `Preprocessor` protocol: `process(audio: np.ndarray, sr: int) -> torch.Tensor` returning shape (1, 1, 128, 64). `Classifier` protocol: `predict(features: torch.Tensor) -> float` returning drone probability [0, 1]. No metadata, no config introspection, no warm-up methods.

### Mel-Spectrogram Library
- **D-06:** Switch from librosa to torchaudio for mel-spectrogram computation. Aligns with v2.0 PyTorch stack, enables GPU acceleration in Phase 8 training pipeline. Parity tests validate against saved reference tensors, so small numerical differences from the library switch are caught and acceptable.

### Parity Test Strategy
- **D-07:** Generate reference tensors by running the TF/librosa research code (`Acoustic-UAV-Identification-main-main/train_strong_cnn.py`) once on known WAV inputs. Save as .npy fixtures committed to repo. PyTorch/torchaudio tests compare against these fixtures with atol=1e-4. No TensorFlow dependency in CI.

### Claude's Discretion
- Where to place `MelConfig` (e.g., `src/acoustic/classification/config.py` or `src/acoustic/config.py`)
- How to restructure `CNNWorker` to accept protocol-injected dependencies
- Which WAV files to use as parity test fixtures (from `audio-data/data/`)
- Whether to keep `fast_resample()` or switch to torchaudio's resampling
- How to handle the `SILENCE_RMS_THRESHOLD` energy gate in the new preprocessor

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Research Reference Implementation
- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` — Canonical research preprocessing: `segment_to_melspec()` (line 73), constants FS=16000, CHUNK_SECONDS=0.5, N_FFT=1024, HOP_LENGTH=256, N_MELS=64, MAX_FRAMES=128, normalization `(S_db+80)/80` clipped [0,1]. This is the ground truth for parity tests.

### Existing Service Code (to be replaced)
- `src/acoustic/classification/preprocessing.py` — Current EfficientNet preprocessing with scattered constants. Replace entirely with research preprocessor.
- `src/acoustic/classification/inference.py` — `OnnxDroneClassifier` to be removed. Replace with Classifier protocol.
- `src/acoustic/classification/worker.py` — `CNNWorker` with direct imports. Refactor to protocol-injected dependencies.
- `src/acoustic/classification/state_machine.py` — `DetectionStateMachine` stays as-is; threshold recalibration is Phase 7.
- `src/acoustic/config.py` — `AcousticSettings` needs CNN config updated (remove ONNX model path, add research config).

### Requirements
- `.planning/REQUIREMENTS.md` — PRE-01, PRE-02, PRE-03, PRE-04 define the acceptance criteria for this phase.

### Project Context
- `.planning/PROJECT.md` — v2.0 milestone goal and constraints.
- `.planning/ROADMAP.md` — Phase 6 success criteria (4 items).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fast_resample()` in `preprocessing.py` — polyphase resampling, may keep or replace with torchaudio equivalent
- `DetectionStateMachine` in `state_machine.py` — unchanged in this phase, Phase 7 recalibrates thresholds
- `CNNWorker` structure in `worker.py` — queue-based background thread pattern reusable after protocol refactor
- `AcousticSettings` in `config.py` — Pydantic BaseSettings pattern for env var config

### Established Patterns
- Config via Pydantic `BaseSettings` with `ACOUSTIC_` env prefix
- Background thread with daemon=True for pipeline processing
- Single-slot queue with drop semantics for non-blocking inference
- FastAPI lifespan for startup/shutdown of background workers

### Integration Points
- `CNNWorker.__init__` — currently takes `OnnxDroneClassifier` directly, needs to accept `Classifier` protocol
- `pipeline.py` — calls `cnn_worker.push()` when peak detected, unchanged interface
- `main.py` — startup factory needs to construct preprocessor + classifier (or None until Phase 7)

</code_context>

<specifics>
## Specific Ideas

- Old EfficientNet pipeline is confirmed non-functional — no need to preserve backward compatibility
- Research normalization `(S_db+80)/80` clipped to [0,1] is the canonical approach (not z-score)
- 0.5s segments everywhere (research standard), replacing the old 2.0s segments
- torchaudio MelSpectrogram transform as the compute engine, not librosa

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-preprocessing-parity-foundation*
*Context gathered: 2026-04-01*
