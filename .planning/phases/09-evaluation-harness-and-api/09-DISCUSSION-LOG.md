# Phase 9: Evaluation Harness and API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 09-evaluation-harness-and-api
**Areas discussed:** Evaluation output, API endpoint design, Training progress streaming, Evaluation execution model

---

## Evaluation Output

### Classification Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Binary only (Recommended) | Matches current ResearchCNN (sigmoid output, BCE loss). Confusion matrix is 2x2. | ✓ |
| Multi-class ready | Design for N-class from the start even though current model is binary. | |
| You decide | Claude picks the approach that best fits the current architecture | |

**User's choice:** Binary only
**Notes:** None

### Per-File Detail Level

| Option | Description | Selected |
|--------|-------------|----------|
| Summary per file (Recommended) | Each file: filename, true label, predicted label, aggregated p_agg score, correct/incorrect. | ✓ |
| Full segment detail | Each file: all segment-level probabilities, plus p_max/p_mean/p_agg. | |
| Both levels | Summary view by default, with option for full segment detail. | |

**User's choice:** Summary per file
**Notes:** None

### Distribution Stats

| Option | Description | Selected |
|--------|-------------|----------|
| Percentiles per class (Recommended) | 25th, 50th, 75th, 95th percentiles of p_agg, p_max, p_mean per class. | ✓ |
| Histograms | Bin probability distributions into histograms for visualization. | |
| You decide | Claude picks whatever gives most actionable threshold-tuning info. | |

**User's choice:** Percentiles per class
**Notes:** None

### Result Persistence

| Option | Description | Selected |
|--------|-------------|----------|
| API only (Recommended) | Results in memory, returned via REST. Ephemeral like training progress. | ✓ |
| Save to JSON file | Write results alongside model checkpoint for historical comparison. | |
| Both | Return via API and write JSON file. | |

**User's choice:** API only
**Notes:** None

---

## API Endpoint Design

### Route Organization

| Option | Description | Selected |
|--------|-------------|----------|
| /api/training + /api/eval (Recommended) | Separate prefixes for training control and evaluation. Clean separation. | ✓ |
| /api/ml/... | Group all ML operations under one prefix. | |
| You decide | Claude picks cleanest organization. | |

**User's choice:** /api/training + /api/eval
**Notes:** None

### Hyperparameter Overrides

| Option | Description | Selected |
|--------|-------------|----------|
| Accept overrides (Recommended) | POST body can include optional lr, batch_size, epochs, patience, augmentation toggle. | ✓ |
| Config-only defaults | No overrides via API -- always use TrainingConfig defaults. | |
| You decide | Claude picks most practical approach. | |

**User's choice:** Accept overrides
**Notes:** None

### Eval Data Path

| Option | Description | Selected |
|--------|-------------|----------|
| Accept path parameter (Recommended) | POST /api/eval/run with optional data_dir in body. | ✓ |
| Fixed directory | Always evaluates against a configured test directory. | |
| You decide | Claude picks most practical. | |

**User's choice:** Accept path parameter
**Notes:** None

### Model Listing Endpoint

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, add GET /api/models | Lists available .pt files with metadata. | ✓ |
| Not needed yet | Phase 9 only deals with current model at cnn_model_path. | |
| You decide | Claude decides based on scope. | |

**User's choice:** Yes, add GET /api/models
**Notes:** None

---

## Training Progress Streaming

### WebSocket Path

| Option | Description | Selected |
|--------|-------------|----------|
| /ws/training (Recommended) | Dedicated WebSocket for training progress. Consistent naming. | ✓ |
| Extend /ws/status | Add training progress to existing /ws/status stream. | |
| You decide | Claude picks cleanest approach. | |

**User's choice:** /ws/training
**Notes:** None

### Push Frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Per-epoch (Recommended) | Push JSON after each epoch with loss, val_loss, val_acc, confusion matrix, status. | ✓ |
| Per-epoch + periodic | Push per-epoch plus heartbeat every few seconds during long epochs. | |
| You decide | Claude decides based on training loop granularity. | |

**User's choice:** Per-epoch
**Notes:** None

### Idle Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Send idle status on connect (Recommended) | One JSON message with current status on connect. Then quiet until training starts. | ✓ |
| Periodic heartbeat | Send status message every N seconds even when idle. | |
| You decide | Claude picks simplest and most useful. | |

**User's choice:** Send idle status on connect
**Notes:** None

---

## Evaluation Execution Model

### Sync vs Async

| Option | Description | Selected |
|--------|-------------|----------|
| Synchronous (Recommended) | POST /api/eval/run blocks and returns results directly. | ✓ |
| Background async | Run in background thread like training. | |
| You decide | Claude picks based on expected duration. | |

**User's choice:** Synchronous
**Notes:** None

### Auto-eval After Training

| Option | Description | Selected |
|--------|-------------|----------|
| On-demand only (Recommended) | User explicitly triggers evaluation. Training just produces checkpoint. | ✓ |
| Auto after training | Automatically run evaluation when training finishes. | |
| Both options | On-demand by default, optional auto_eval flag in training start request. | |

**User's choice:** On-demand only
**Notes:** None

### Model Source for Evaluation

| Option | Description | Selected |
|--------|-------------|----------|
| Accept model path (Recommended) | POST body includes optional model_path. Defaults to cnn_model_path. | ✓ |
| Always use live model | Evaluate whatever model is currently loaded. | |
| You decide | Claude picks most flexible. | |

**User's choice:** Accept model path
**Notes:** None

---

## Additional User Input

**Test/Training Data:** User specified that `Acoustic-UAV-Identification-main-main/Recorded Audios/` should be used as labeled test data for evaluation AND as training data. Directory structure: `Real World Testing/drone/` (272 files), `Real World Testing/no drone/` (284 files), plus `Unseen Drone Audio/` (2 additional drone WAVs).

## Claude's Discretion

- Evaluation module structure placement
- Pydantic request/response models
- Model file scanning logic for `/api/models`
- Route file organization (new files vs extending existing)
- Label mapping for folder names with spaces (`no drone` -> background/not-drone)

## Deferred Ideas

None -- discussion stayed within phase scope
