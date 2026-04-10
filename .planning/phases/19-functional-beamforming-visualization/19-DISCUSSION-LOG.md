# Phase 19: Functional Beamforming Visualization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 19-functional-beamforming-visualization
**Areas discussed:** Functional beamforming algorithm, Nu parameter control, Frequency band wiring, Heatmap visual tuning

---

## Functional Beamforming Algorithm

| Option | Description | Selected |
|--------|-------------|----------|
| Power-map exponent | Raise normalized SRP-PHAT map to nu-th power. Simple post-processing, no engine changes. | ✓ |
| CSM eigenvalue decomposition | True functional beamforming per Dougherty (2014). More accurate but high compute cost. | |
| You decide | Claude picks based on constraints. | |

**User's choice:** Power-map exponent (Recommended)
**Notes:** Standard approach for functional beamforming visualization. Fast and well-understood.

### Follow-up: Transform Placement

| Option | Description | Selected |
|--------|-------------|----------|
| Backend pipeline | Apply in Python pipeline before WebSocket send. Frontend stays dumb renderer. | ✓ |
| Frontend transform | Send raw map, apply in JavaScript. Requires frontend to know nu value. | |
| You decide | Claude picks based on architecture. | |

**User's choice:** Backend pipeline (Recommended)
**Notes:** Consistent with current architecture where pipeline owns all processing.

---

## Nu Parameter Control

| Option | Description | Selected |
|--------|-------------|----------|
| Config + REST API | Add bf_nu to settings + PATCH endpoint for runtime updates. | ✓ |
| Config + UI slider | Same plus slider in Monitor page. More frontend work. | |
| Config only | Env var only, requires restart. | |
| You decide | Claude picks. | |

**User's choice:** Config + REST API (Recommended)
**Notes:** Matches existing config pattern for other beamforming parameters.

---

## Frequency Band Wiring

| Option | Description | Selected |
|--------|-------------|----------|
| Update defaults | Change srp_phat_2d() defaults from 100/2000 to 500/4000. | ✓ |
| Leave as-is | Pipeline always passes explicit values so defaults don't matter. | |
| You decide | Claude picks. | |

**User's choice:** Update defaults (Recommended)
**Notes:** Pipeline already passes bf_freq_min=500/bf_freq_max=4000 correctly. Update defaults for consistency.

---

## Heatmap Visual Tuning

| Option | Description | Selected |
|--------|-------------|----------|
| Backend normalizes to [0,1] | Backend applies map^nu then normalizes. Frontend renders directly. | ✓ |
| Log-scale post-processing | Apply log scaling after nu transform for mid-range detail. | |
| You decide | Claude picks. | |

**User's choice:** Backend normalizes to [0,1] (Recommended)

### Follow-up: Frontend v² Squaring

| Option | Description | Selected |
|--------|-------------|----------|
| Remove v² entirely | Backend owns all contrast. Frontend just maps [0,1] to colors. | ✓ |
| Conditional on nu | Keep v² when nu=1, remove when nu>1. More complex. | |
| You decide | Claude picks. | |

**User's choice:** Remove v² entirely (Recommended)
**Notes:** Simpler single code path. Backend owns normalization.

---

## Claude's Discretion

- Exact normalization strategy (min-max vs max-only)
- Whether to clamp small values to zero
- PATCH settings endpoint implementation details

## Deferred Ideas

None — discussion stayed within phase scope.
