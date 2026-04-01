# Phase 8: PyTorch Training Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 08-pytorch-training-pipeline
**Areas discussed:** Dataset & data loading, Training loop & early stopping, Data augmentation, Resource isolation & background execution

---

## Dataset & Data Loading

### Segment Extraction Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Random 0.5s segments | Each epoch picks a random 0.5s window per WAV file — matches research approach, implicit augmentation | ✓ |
| Multiple fixed segments per file | Split each WAV into all possible 0.5s segments, more data points but longer epochs | |
| You decide | Claude picks based on research reference | |

**User's choice:** Random 0.5s segments (Recommended)

### Class Balancing

| Option | Description | Selected |
|--------|-------------|----------|
| Weighted sampling | WeightedRandomSampler over-samples minority class, uses all data | ✓ |
| Undersample majority | Cap majority to match minority count, discards data | |
| You decide | Claude picks | |

**User's choice:** Weighted sampling (Recommended)

### Train/Validation Split

| Option | Description | Selected |
|--------|-------------|----------|
| 80/20 | Standard split, shuffled at file level | ✓ |
| 85/15 | More training data, smaller validation set | |
| You decide | Claude picks | |

**User's choice:** 80/20 (Recommended)

### Data Directory Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Existing structure | Scan audio-data/data/{drone,background,other}/ | |
| Configurable root dir | Accept any directory with label subdirectories via config | ✓ |
| You decide | Claude picks | |

**User's choice:** Configurable root dir

---

## Training Loop & Early Stopping

### Early Stopping Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Patience-based on val loss | Stop after N epochs with no improvement, save best checkpoint, patience=5 | ✓ |
| Fixed epoch count | Always train for N epochs | |
| You decide | Claude picks | |

**User's choice:** Patience-based on val loss (Recommended)

### Default Hyperparameters

| Option | Description | Selected |
|--------|-------------|----------|
| Research defaults | Adam, lr=1e-3, batch_size=32, max_epochs=50, patience=5 | ✓ |
| Conservative defaults | Adam, lr=1e-4, batch_size=16, max_epochs=30, patience=10 | |
| You decide | Claude picks | |

**User's choice:** Research defaults (Recommended)

### Export Format

| Option | Description | Selected |
|--------|-------------|----------|
| PyTorch .pt only | Save state_dict as .pt, matches Phase 7 D-05 | ✓ |
| PyTorch .pt + ONNX | Save both formats | |
| You decide | Claude picks | |

**User's choice:** PyTorch .pt only (Recommended)

---

## Data Augmentation

### SpecAugment Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Time + frequency masking | Standard SpecAugment: time masks (up to 20 frames) + frequency masks (up to 8 mel bins) | ✓ |
| Frequency masking only | Skip time masking, simpler | |
| You decide | Claude picks | |

**User's choice:** Time + frequency masking (Recommended)

### Waveform Augmentation

| Option | Description | Selected |
|--------|-------------|----------|
| Noise + gain only | Gaussian noise (SNR 10-40dB) + random gain (±6dB) | ✓ |
| Noise + gain + time stretch | Also add time stretching (0.8-1.2x) | |
| None | Skip waveform augmentation | |
| You decide | Claude picks | |

**User's choice:** Noise + gain only (Recommended)

### Augmentation Toggle

| Option | Description | Selected |
|--------|-------------|----------|
| Single toggle | One config flag for all augmentation on/off | ✓ |
| Per-augmentation toggles | Separate flags for each augmentation type | |
| You decide | Claude picks | |

**User's choice:** Single toggle (Recommended)

---

## Resource Isolation & Background Execution

### Background Execution Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Background thread with os.nice | Daemon thread with os.nice(10) + torch.set_num_threads(2) | ✓ |
| Subprocess | Separate Python process, full isolation but complex IPC | |
| You decide | Claude picks | |

**User's choice:** Background thread with os.nice (Recommended)

### Progress Reporting

| Option | Description | Selected |
|--------|-------------|----------|
| In-memory state object | Thread-safe dataclass polled by API endpoints | ✓ |
| Log file + state object | File persistence plus state object | |
| You decide | Claude picks | |

**User's choice:** In-memory state object (Recommended)

### Cancellation

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, via stop flag | Check threading event between epochs, preserve best checkpoint | ✓ |
| No cancellation | Run to completion or early stopping | |
| You decide | Claude picks | |

**User's choice:** Yes, via stop flag (Recommended)

### Concurrency

| Option | Description | Selected |
|--------|-------------|----------|
| Single run only | One training job at a time, error if already running | ✓ |
| Queue-based | Sequential job queue | |
| You decide | Claude picks | |

**User's choice:** Single run only (Recommended)

---

## Claude's Discretion

- Module placement (src/acoustic/training/ vs classification/training.py)
- PyTorch Dataset implementation details
- Learning rate scheduler choice
- SpecAugment exact parameters
- Thread-safe state object implementation
- Preprocessor reuse strategy

## Deferred Ideas

None — discussion stayed within phase scope
