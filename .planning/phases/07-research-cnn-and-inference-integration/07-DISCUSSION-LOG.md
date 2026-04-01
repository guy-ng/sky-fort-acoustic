# Phase 7: Research CNN and Inference Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 07-research-cnn-and-inference-integration
**Areas discussed:** Segment aggregation strategy, Model loading & factory wiring, State machine recalibration, Aggregator protocol design

---

## Segment Aggregation Strategy

### Q1: How much audio should be aggregated before making a detection decision?

| Option | Description | Selected |
|--------|-------------|----------|
| 2 seconds (4 segments) | Matches old 2.0s CNN window. 4 overlapping 0.5s segments. Balances latency vs. stability. | ✓ |
| 1 second (2 segments) | Faster response, less smoothing. May increase false positives. | |
| 3 seconds (6 segments) | Maximum stability, slower reaction. Better for steady surveillance. | |

**User's choice:** 2 seconds (4 segments)
**Notes:** None

### Q2: How should segment predictions be combined into a final probability?

| Option | Description | Selected |
|--------|-------------|----------|
| Weighted combo | p_agg = w1*p_max + w2*p_mean with configurable weights (default 0.5/0.5). | ✓ |
| Max only (p_max) | Highest segment probability. Most sensitive. Risk of false positives. | |
| Mean only (p_mean) | Average across segments. Most conservative. May miss brief fly-bys. | |

**User's choice:** Weighted combo
**Notes:** None

### Q3: Should segments overlap, and by how much?

| Option | Description | Selected |
|--------|-------------|----------|
| 50% overlap | 0.25s hop between segments. Standard in audio ML. | ✓ |
| No overlap | Simpler buffering but events at boundaries may be missed. | |
| You decide | Let Claude choose based on research code and latency constraints. | |

**User's choice:** 50% overlap
**Notes:** None

---

## Model Loading & Factory Wiring

### Q1: What should happen when the configured model file doesn't exist at startup?

| Option | Description | Selected |
|--------|-------------|----------|
| Start without classifier | Service boots normally, CNNWorker dormant. Logs warning. Beamforming-only operation. | ✓ |
| Fail fast | Service refuses to start if model file missing. | |
| You decide | Let Claude choose based on operational patterns. | |

**User's choice:** Start without classifier
**Notes:** None

### Q2: Should the factory support loading both .pt and .h5 model files?

| Option | Description | Selected |
|--------|-------------|----------|
| PyTorch .pt only | Clean break. Only native PyTorch state_dict loading. | ✓ |
| Both formats | Auto-detect from extension. Allows legacy .h5 during transition. | |
| You decide | Let Claude choose based on v2.0 migration goals. | |

**User's choice:** PyTorch .pt only
**Notes:** None

---

## State Machine Recalibration

### Q1: How should Phase 7 handle the unknown confidence distribution of the new CNN?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep defaults, make configurable | Ship with 0.80/0.40/2. Already env-var configurable. Phase 9 determines optimal values. | ✓ |
| Lower initial thresholds | Drop to enter=0.60, exit=0.30 as safer starting point. | |
| You decide | Let Claude choose based on research paper confidence distributions. | |

**User's choice:** Keep defaults, make configurable
**Notes:** None

### Q2: Should the aggregated probability feed the state machine directly?

| Option | Description | Selected |
|--------|-------------|----------|
| Direct feed | Aggregated p_agg feeds directly as single probability. No SM changes needed. | ✓ |
| Per-segment + aggregated | SM receives both individual and aggregated values. More complex. | |
| You decide | Let Claude determine cleanest integration. | |

**User's choice:** Direct feed
**Notes:** None

---

## Aggregator Protocol Design

### Q1: Should aggregation be a formal protocol or internal to CNNWorker?

| Option | Description | Selected |
|--------|-------------|----------|
| Protocol | New Aggregator protocol: aggregate(probabilities) -> float. Injected into CNNWorker. Enables Phase 11 ensemble swaps. | ✓ |
| Internal to CNNWorker | Private method inside worker. Simpler now, harder to swap later. | |
| You decide | Let Claude choose based on existing protocol pattern. | |

**User's choice:** Protocol
**Notes:** None

### Q2: Should aggregation weights be configurable via environment variables?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, via env vars | ACOUSTIC_CNN_AGG_W_MAX and ACOUSTIC_CNN_AGG_W_MEAN in AcousticSettings. Defaults 0.5/0.5. | ✓ |
| Hardcoded defaults only | Weights fixed at 0.5/0.5 in code. Change requires code edit. | |
| You decide | Let Claude choose based on existing config pattern. | |

**User's choice:** Yes, via env vars
**Notes:** None

---

## Claude's Discretion

- PyTorch model class placement
- Segment buffer implementation in CNNWorker
- Model validation on load
- Overlapping segment generation from audio buffer
- Default Aggregator implementation class name and location

## Deferred Ideas

None -- discussion stayed within phase scope
