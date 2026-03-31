# Phase 3: CNN Classification and Target Tracking - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-31
**Phase:** 03-cnn-classification-and-target-tracking
**Areas discussed:** CNN Architecture & Inference, Target Tracking, ZeroMQ Events

---

## Scope Narrowing

User immediately clarified that drone type classification and Doppler speed estimation are deferred to milestone 2. Phase 3 is binary drone/not-drone detection only.

---

## CNN Source

| Option | Description | Selected |
|--------|-------------|----------|
| POC's trained model | Port the POC's CNN detection approach, retrain with PyTorch | |
| Beamforming peak only | Skip CNN, use beamforming peak as detection signal | |
| Pre-trained audio model | Fine-tune YAMNet/PANNs on drone data | |

**User's choice:** POC's trained model — but use it as-is, not retrain
**Notes:** User wants to reuse the existing trained .h5 model, not rebuild from scratch

---

## CNN Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Binary only (Recommended) | Single sigmoid output, drone probability | ✓ |
| Binary + confidence calibration | Add Platt/temperature scaling | |

**User's choice:** Binary only
**Notes:** None

---

## Training Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Build training in Phase 5 | Ship with pre-trained model, training comes later | |
| Quick training script now | Minimal PyTorch training script in Phase 3 | |

**User's choice:** Use the existing POC model directly, build training script later
**Notes:** User asked if they can use the model trained in POC — yes, via ONNX conversion

---

## Model Format

| Option | Description | Selected |
|--------|-------------|----------|
| Convert to ONNX (Recommended) | One-time .h5 to .onnx conversion, inference via onnxruntime | ✓ |
| Keep TF for inference | Load .h5 directly with tensorflow/keras | |
| Reimplement in PyTorch | Recreate architecture, transfer weights | |

**User's choice:** Convert to ONNX
**Notes:** Keeps Docker image lean (~50MB vs ~500MB for TF)

---

## CNN Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| Gate on beamforming peak (Recommended) | Only run CNN when peak detected | ✓ |
| Run CNN continuously | Process every 2s window regardless | |

**User's choice:** Gate on beamforming peak
**Notes:** Matches POC's ENERGY+MODEL gating pattern

---

## Target TTL

| Option | Description | Selected |
|--------|-------------|----------|
| 5 seconds (Recommended) | Target stays alive 5s after last detection | ✓ |
| 10 seconds | Longer persistence for intermittent signals | |
| You decide | Claude picks default, configurable via env var | |

**User's choice:** 5 seconds
**Notes:** None

---

## ZMQ Topic Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single topic, event types | Topic: "acoustic/targets", event field: new/update/lost | ✓ |
| Separate topics per event | Topics: acoustic/target/new, /update, /lost | |
| You decide | Claude picks based on ZMQ best practices | |

**User's choice:** Single topic, event types
**Notes:** None

---

## Claude's Discretion

- Hysteresis thresholds and confirmation hit counts
- CNN worker threading model
- ZMQ publish frequency for update events
- Pipeline integration architecture
- Multi-target handling
- Web UI updates for real detection data

## Deferred Ideas

- Drone type classification (CLS-02) — deferred to milestone 2
- Doppler speed estimation (TRK-02) — deferred to milestone 2
- CNN training pipeline — Phase 5 as planned
